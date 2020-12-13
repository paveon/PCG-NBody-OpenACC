/**
 * @file      nbody.cpp
 *
 * @author    Ondřej Pavela - xpavel34 \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            xpavel34@stud.fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *            N-Body simulation in ACC
 *
 * @version   2021
 *
 * @date      11 November  2020, 11:22 (created) \n
 * @date      11 November  2020, 11:37 (revised) \n
 *
 */

#include <math.h>
#include <cfloat>
#include <chrono>
#include "nbody.h"


/// Compute gravitation velocity
void calculate_velocity(const Particles& p_in, Particles& p_out, const int N, const float dt, unsigned streamID)
{
  #pragma acc parallel loop vector_length(1024) present(p_in, p_out) async(streamID)
  for (unsigned p1_idx = 0; p1_idx < N; p1_idx++) {
    vec3 accVelocity;
    vec4 p1 = p_in.GetPosition(p1_idx);
    vec3 p1_vel = p_in.GetVelocity(p1_idx);

    #pragma acc loop seq
    for (unsigned p2_idx = 0; p2_idx < N; p2_idx++) {
      vec4 p2 = p_in.GetPosition(p2_idx);
      vec3 delta = p1.toVec3() - p2.toVec3();
      float rr = delta.LengthSq();
      float r = sqrtf(rr);

      if (r > COLLISION_DISTANCE) {
          float r3 = rr * r + FLT_MIN;
          float G_dt_r3 = -G * dt / r3;
          float Fg_dt_m2_r = G_dt_r3 * p2.w;
          accVelocity += Fg_dt_m2_r * delta;
      }
      else {
        vec3 p2_vel = p_in.GetVelocity(p2_idx);
        float weightSum = p1.w + p2.w;
        float weightDiff = p1.w - p2.w;
        float p2_w2 = 2 * p2.w;
        if (r > 0.0f) {
          accVelocity += (((p1_vel * weightDiff) + (p2_w2 * p2_vel)) / weightSum) - p1_vel;
        }
      }
    }

    vec3 newVelocity = p1_vel + accVelocity;
    p_out.SetPosition(p1_idx, p1.toVec3() + (newVelocity * dt));
    p_out.SetVelocity(p1_idx, newVelocity);
  }
}// end of calculate_velocity
//----------------------------------------------------------------------------------------------------------------------


/// Compute center of gravity
// void centerOfMassGPU(const Particles& p, vec4* tmp_com, const unsigned N, unsigned streamID)
// {
//   size_t nearestN = powf(2, ceilf(log2f(N)));

//   #pragma acc parallel loop present(p, tmp_com[N]) async(streamID)
//   for (size_t i = 0; i < N; i++) {
//     vec4 particle = p.GetPosition(i);
//     tmp_com[i] = (particle.w > 0.0f) ? particle : vec4(0.0f);
//   }

//   for (size_t stride = nearestN >> 1ul; stride > 32; stride >>= 1ul) {
//     #pragma acc parallel loop gang vector_length(512) present(p, tmp_com[N]) async(streamID)
//     for (size_t i = 0; i < stride; i++) {
//       if (i + stride < N) {
//         vec4 com_1 = tmp_com[i];
//         vec4 com_2 = tmp_com[i + stride];
//         vec3 delta = com_2.toVec3() - com_1.toVec3();
//         float dw = ((com_2.w + com_1.w) > 0.0f) ? (com_2.w / (com_2.w + com_1.w)) : 0.0f;
//         tmp_com[i] += delta * dw;
//         tmp_com[i].w += com_2.w;
//       }
//     }
//   }

//   #pragma acc parallel loop worker num_gangs(1) num_workers(1) vector_length(32) present(p, tmp_com[N]) async(streamID)
//   for (size_t stride = 32; stride > 0; stride >>= 1ul) {
//     #pragma acc loop vector
//     for (size_t i = 0; i < stride; i++) {
//       if (i + stride < N) {
//         vec4 com_1 = tmp_com[i];
//         vec4 com_2 = tmp_com[i + stride];
//         vec3 delta = com_2.toVec3() - com_1.toVec3();
//         float dw = ((com_2.w + com_1.w) > 0.0f) ? (com_2.w / (com_2.w + com_1.w)) : 0.0f;
//         tmp_com[i] += delta * dw;
//         tmp_com[i].w += com_2.w;
//       }
//     }
//   }
// }// end of centerOfMassGPU



/// Compute center of gravity
size_t centerOfMassGPU2(const Particles& p, vec4* __restrict tmp_com_in, vec4* __restrict tmp_com_out, const unsigned N, unsigned streamID)
{
  const size_t chunkSize = 64;
  size_t steps = ceilf(log2f(N) / 6.0f); /// ceil(log64(N))

  #pragma acc parallel loop gang vector_length(1024) present(p, tmp_com_in[N]) async(streamID)
  for (size_t i = 0; i < N; i++) {
    vec4 particle = p.GetPosition(i);
    tmp_com_in[i] = (particle.w > 0.0f) ? particle : vec4(0.0f);
  }

  vec4* doubleBuffer[2] = {
    tmp_com_in,
    tmp_com_out
  };

  /// Run the kernel log64(N) times
  size_t inputSize = N;
  for (size_t step = 0; step < steps; step++) {
    size_t outputSize = ceilf((float)inputSize / (float)chunkSize);

    vec4* __restrict in = doubleBuffer[step & 1ul];
    vec4* __restrict out = doubleBuffer[(step + 1) & 1ul];

    /// Each block has a single warp which reduces 64 values and stores the result into an
    /// output buffer. Double buffering is used to avoid sparse result array of each reduction level
    #pragma acc parallel loop gang num_workers(1) vector_length(32) present(p, in[inputSize], out[outputSize]) async(streamID)
    for (size_t chunkIdx = 0; chunkIdx < outputSize; chunkIdx++) {
      unsigned chunkOffset = chunkIdx * chunkSize;

      #pragma acc loop worker
      for (size_t stride = 32; stride > 0; stride >>= 1ul) {

        #pragma acc loop vector
        for (size_t threadIdx = 0; threadIdx < stride; threadIdx++) {

          unsigned threadOffset = chunkOffset + threadIdx;
          if (threadOffset + stride < inputSize) {
            vec4 com_1 = in[threadOffset];
            vec4 com_2 = in[threadOffset + stride];
            vec3 delta = com_2.toVec3() - com_1.toVec3();
            float dw = ((com_2.w + com_1.w) > 0.0f) ? (com_2.w / (com_2.w + com_1.w)) : 0.0f;

            in[threadOffset] += delta * dw;
            in[threadOffset].w += com_2.w;
          }
        }
      }

      out[chunkIdx] = in[chunkOffset];
    }

    inputSize = outputSize;
  }

  return (steps & 1ul);
}// end of centerOfMassGPU


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute center of mass on CPU
vec4 centerOfMassCPU(MemDesc& memDesc)
{
  vec4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// end of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
