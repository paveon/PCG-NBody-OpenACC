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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute gravitation velocity
void calculate_gravitation_velocity(const Particles& p, Velocities& tmp_vel, const int N, const float dt)
{
  #pragma acc parallel loop vector_length(1024) present(p, tmp_vel)
  for (unsigned p1_idx = 0; p1_idx < N; p1_idx++) {
    vec3 accVelocity;
    vec4 p1 = p.GetPosition(p1_idx);

    #pragma acc loop seq
    for (unsigned p2_idx = 0; p2_idx < N; p2_idx++) {
      vec4 p2 = p.GetPosition(p2_idx);
      vec3 delta = p1.toVec3() - p2.toVec3();
      float rr(delta.LengthSq());
      float r = sqrtf(rr);
      
      float F = -G * p1.w * p2.w / (rr + FLT_MIN);
      float dtw = dt / p1.w;

      if (r > COLLISION_DISTANCE) {
        accVelocity += (F * delta / (r + FLT_MIN)) * dtw;
      }
    }

    tmp_vel.AddVelocity(p1_idx, accVelocity);
  }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

void calculate_collision_velocity(const Particles& p, Velocities& tmp_vel, const int N, const float dt) 
{
  #pragma acc parallel loop vector_length(1024) present(p, tmp_vel)
  for (unsigned p1_idx = 0; p1_idx < N; p1_idx++) {
    vec3 accVelocity;
    vec4 p1 = p.GetPosition(p1_idx);
    vec3 p1_vel = p.GetVelocity(p1_idx);

    #pragma acc loop seq
    for (unsigned p2_idx = 0; p2_idx < N; p2_idx++) {
      vec4 p2 = p.GetPosition(p2_idx);
      vec3 p2_vel = p.GetVelocity(p2_idx);
      vec3 delta = p1.toVec3() - p2.toVec3();
      float rr = delta.LengthSq();
      float r = sqrtf(rr);

      float weightSum = p1.w + p2.w;
      float weightDiff = p1.w - p2.w;
      float p2_w2 = 2 * p2.w;

      if (r > 0.0f && r < COLLISION_DISTANCE) {
        accVelocity += (((p1_vel * weightDiff) + (p2_w2 * p2_vel)) / weightSum) - p1_vel;
      }
    }

    tmp_vel.AddVelocity(p1_idx, accVelocity);
  }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/// Update particle position
void update_particle(Particles& p, Velocities& tmp_vel, const int N, const float dt)
{
  #pragma acc parallel loop vector_length(1024) present(p, tmp_vel)
  for (unsigned i = 0; i < N; i++) {
    vec3 newVelocity = p.GetVelocity(i) + tmp_vel.GetVelocity(i);
    p.SetVelocity(i, newVelocity);
    p.MovePosition(i, newVelocity * dt);
  }
}// end of update_particle
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
