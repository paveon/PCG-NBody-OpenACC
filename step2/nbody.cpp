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
#include "nbody.h"


/// Compute gravitation velocity
void calculate_velocity(const Particles& p_in, Particles& p_out, const int N, const float dt)
{
  #pragma acc parallel loop vector_length(1024) present(p_in, p_out)
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
vec4 centerOfMassGPU(const Particles& p,
                       const int        N)
{

  return {0.0f, 0.0f, 0.0f, 0.0f};
}// end of centerOfMassGPU
//----------------------------------------------------------------------------------------------------------------------

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
