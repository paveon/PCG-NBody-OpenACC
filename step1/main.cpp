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

#include "nbody.h"
#include "h5Helper.h"


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

  const int   N         = std::stoi(argv[1]);
  const float dt        = std::stof(argv[2]);
  const int   steps     = std::stoi(argv[3]);
  const int   writeFreq = (std::stoi(argv[4]) > 0) ? std::stoi(argv[4]) : 0;

  printf("N: %d\n", N);
  printf("dt: %f\n", dt);
  printf("steps: %d\n", steps);

  const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;

  Particles particles(N);
  Velocities velocities(N);

  MemDesc md(
                   &particles.m_Positions[0].x, 4, 0,
                   &particles.m_Positions[0].y, 4, 0,
                   &particles.m_Positions[0].z, 4, 0,
                   &particles.m_Velocities[0].x, 3, 0,
                   &particles.m_Velocities[0].y, 3, 0,
                   &particles.m_Velocities[0].z, 3, 0,
                   &particles.m_Positions[0].w, 4, 0,
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

  particles.updateDevice();
  velocities.Memset(0);


  // Start the time
  auto startTime = std::chrono::high_resolution_clock::now();

  for (int s = 0; s < steps; s++)
  {
    calculate_gravitation_velocity(particles, velocities, N, dt);
    calculate_collision_velocity(particles, velocities, N, dt);
    update_particle(particles, velocities, N, dt);

    velocities.MemsetDeviceMemory(0);
  }

  vec4 comOnGPU = {0.0f, 0.0f, 0.0f, 0.f};

  // Stop watchclock
  const auto   endTime = std::chrono::high_resolution_clock::now();
  const double time    = (endTime - startTime) / std::chrono::milliseconds(1);
  printf("Time: %f s\n", time / 1000);


  particles.updateHost();

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

  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

