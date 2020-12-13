/**
 * @file      main.cpp
 *
 * @author    Ondřej Pavela - xpavel34
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            xpavel34@stud.fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *            N-Body simulation in ACC
 *
 * @version   2021
 *
 * @date      11 November  2020, 11:22 (created)
 * @date      11 November  2020, 11:37 (revised)
 *
 */

#include <chrono>
#include <cstdio>
#include <cmath>
#include <array>

#include "nbody.h"
#include "h5Helper.h"

const unsigned velocityComputeStream = 1;
const unsigned comComputeStream = 2;
const unsigned particleTransferStream = 3;
const unsigned comTransferStream = 4;


/**
 * Main routine of the project
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  // Parse command line parameters
  if (argc != 7)
  {
    printf("Usage: nbody <N> <dt> <steps> <write intesity> <input> <output>\n");
    exit(EXIT_FAILURE);
  }

  const int N = std::stoi(argv[1]);
  const float dt = std::stof(argv[2]);
  const size_t steps = std::stoi(argv[3]);
  const int writeFreq = (std::stoi(argv[4]) > 0) ? std::stoi(argv[4]) : 0;

  printf("N: %d\n", N);
  printf("dt: %f\n", dt);
  printf("steps: %d\n", steps);

  const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;

  Particles particles_in(N);

  MemDesc md(
                   &particles_in.m_Positions[0].x, 4, 0,
                   &particles_in.m_Positions[0].y, 4, 0,
                   &particles_in.m_Positions[0].z, 4, 0,
                   &particles_in.m_Velocities[0].x, 3, 0,
                   &particles_in.m_Velocities[0].y, 3, 0,
                   &particles_in.m_Velocities[0].z, 3, 0,
                   &particles_in.m_Positions[0].w, 4, 0,
                   N,
                   recordsNum);

  H5Helper h5Helper(argv[5], argv[6], md);

  // Read data
  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  } catch (const std::exception& e)
  {
    std::cerr<<e.what()<<std::endl;
    return EXIT_FAILURE;
  }

  particles_in.updateDevice();
  Particles particles_out(particles_in);

  std::vector<Particles*> particles{ 
    &particles_in, 
    &particles_out
  };

  // size_t nearestN = powf(2, ceilf(log2f(N)));
  size_t chunkCount = ceilf((float)N / (float)64.0f);
  std::array<std::vector<vec4>, 2> doubleBufferCOM {
    std::vector<vec4>(N),
    std::vector<vec4>(chunkCount), 
  };
  #pragma acc enter data create(doubleBufferCOM[0].data()[N])
  #pragma acc enter data create(doubleBufferCOM[1].data()[chunkCount])

  // Start the time
  auto startTime = std::chrono::high_resolution_clock::now();
  size_t recordID = 0;

  for (size_t s = 0; s < steps; s++)
  {
    // centerOfMassGPU(*particles[s & 1ul], tmp_com.data(), N, comComputeStream);
    size_t bufferIdx = centerOfMassGPU2(*particles[s & 1ul], 
                                        doubleBufferCOM[0].data(),
                                        doubleBufferCOM[1].data(),
                                        N,
                                        comComputeStream);

    if (writeFreq > 0 && (s % writeFreq == 0))
    {
      if (s > 0) {
        /// Wait for the particle computation from the previous iteration to finish
        particles[s & 1ul]->updateHostAsync(particleTransferStream, velocityComputeStream);
      }

      /// Wait for the COM computation from the current iteration to finish
      #pragma acc update host(doubleBufferCOM[bufferIdx].data()[1]) async(comTransferStream) wait(comComputeStream)

      /// Wait for the particle data transfer to finish
      #pragma acc wait(particleTransferStream)

      /// h5Helper reads data only from the first particle data structure so we need to copy
      /// the data if the second one was chosen as the output destination of the previous iteration
      if (steps & 1ul != 0) {
        std::memcpy(particles[0]->m_Positions, particles[1]->m_Positions, sizeof(vec4) * N);
        std::memcpy(particles[0]->m_Velocities, particles[1]->m_Velocities, sizeof(vec3) * N);
      }
      h5Helper.writeParticleData(recordID);
      
      /// Wait for the COM data transfer to finish
      #pragma acc wait(comTransferStream)
      vec4 comOnGPU = doubleBufferCOM[bufferIdx][0];
      h5Helper.writeCom(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w, recordID);

      recordID++;
    }

    calculate_velocity(*particles[s & 1ul], *particles[(s + 1) & 1ul], N, dt, velocityComputeStream);

    /// Synchronize both compute streams across loop iterations, i.e.,
    /// both compute streams must wait for each other to finish current iteration
    /// This should basically translate to record/wait events between those two streams
    #pragma acc wait(velocityComputeStream) async(comComputeStream)
    #pragma acc wait(comComputeStream) async(velocityComputeStream)
  }
  // #pragma acc wait(velocityComputeStream, comComputeStream)
  
  size_t bufferIdx = centerOfMassGPU2(*particles[steps & 1ul],
                                      doubleBufferCOM[0].data(),
                                      doubleBufferCOM[1].data(),
                                      N,
                                      comComputeStream);

  #pragma acc update host(doubleBufferCOM[bufferIdx].data()[1]) wait(comComputeStream)
  vec4 comOnGPU = doubleBufferCOM[bufferIdx][0];

  // Stop watchclock
  const auto   endTime = std::chrono::high_resolution_clock::now();
  const double time    = (endTime - startTime) / std::chrono::milliseconds(1);
  printf("Time: %f s\n", time / 1000);

  particles[steps & 1ul]->updateHost();
  if ((steps & 1ul) > 0) {
    std::memcpy(particles[0]->m_Positions, particles[1]->m_Positions, sizeof(vec4) * N);
    std::memcpy(particles[0]->m_Velocities, particles[1]->m_Velocities, sizeof(vec3) * N);
  }

  /// Calculate center of gravity
  vec4 comOnCPU = centerOfMassCPU(md);

  std::cout<<"Center of mass on CPU:"<<std::endl
    << comOnCPU.x <<", "
    << comOnCPU.y <<", "
    << comOnCPU.z <<", "
    << comOnCPU.w
    << std::endl;

  std::cout<<"Center of mass on GPU:"<<std::endl
    << comOnGPU.x <<", "
    << comOnGPU.y <<", "
    << comOnGPU.z <<", "
    << comOnGPU.w
    <<std::endl;

  // Store final positions of the particles into a file
  h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
  h5Helper.writeParticleDataFinal();

  #pragma acc exit data delete(doubleBufferCOM[0].data()[N])
  #pragma acc exit data delete(doubleBufferCOM[1].data()[chunkCount])  

  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

