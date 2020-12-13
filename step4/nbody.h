/**
 * @file      nbody.h
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

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "h5Helper.h"
#include "particles.h"

/**
 * Compute gravitation velocity
 * @param [in]  p        - Particles
 * @param [out] tmp_vel  - Temporal velocity
 * @param [in ] N        - Number of particles
 * @param [in]  dt       - Time step size
 */
void calculate_velocity(const Particles&  p_in,
                        Particles&        p_out,
                        const int         N,
                        const float       dt,
                        unsigned          streamID);

/**
 * Compute center of gravity - implement in steps 3 and 4.
 * @param [in] p - Particles
 * @param [in] N - Number of particles
 * @return Center of Mass [x, y, z] and total weight[w]
 */
// void centerOfMassGPU(const Particles& p, vec4* tmp_com, const unsigned N, unsigned streamID);

size_t centerOfMassGPU2(const Particles& p, 
                      vec4* __restrict tmp_com_in,
                      vec4* __restrict tmp_com_out,
                      const unsigned N,
                      unsigned streamID);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Compute center of mass on CPU
 * @param memDesc
 * @return centre of gravity
 */
vec4 centerOfMassCPU(MemDesc& memDesc);

#endif /* __NBODY_H__ */
