/**
 * @file      particles.h
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

#ifndef __PARTICLES_H__
#define __PARTICLES_H__

#include <cstdio>
#include <cmath>
#include <cstring>
#include <iostream>


/// Gravity constant
constexpr float G = 6.67384e-11f;

/// Collision distance threshold
constexpr float COLLISION_DISTANCE = 0.01f;

/// Define sqrtf from CUDA libm library
#pragma acc routine(sqrtf) seq

inline size_t roundUp(size_t number, size_t multiple) {
    return (number + multiple - 1) & ~(multiple - 1);
}


struct vec3 {
  float x, y, z;

  vec3() : x(0.0f), y(0.0f), z(0.0f) {}

  vec3(float value) : x(value), y(value), z(value) {}

  vec3(float x, float y, float z) : x(x), y(y), z(z) {}

  #pragma acc routine seq
  vec3& operator+=(const vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  #pragma acc routine seq
  vec3 operator-(const vec3& rhs) const {
    return { x - rhs.x, y - rhs.y, z - rhs.z };
  }

  #pragma acc routine seq
  vec3 operator+(const vec3& rhs) const {
    return { x + rhs.x, y + rhs.y, z + rhs.z };
  }

  #pragma acc routine seq
  vec3 operator*(const vec3& rhs) const {
    return { x * rhs.x, y * rhs.y, z * rhs.z };
  }

  #pragma acc routine seq
  vec3 operator/(float value) const {
    return { x / value, y / value, z / value };
  }

  #pragma acc routine seq
  float LengthSq() const { return (x * x) + (y * y) + (z * z); }
};

#pragma acc routine seq
inline vec3 operator*(const vec3& lhs, float x) {
  return { lhs.x * x, lhs.y * x, lhs.z * x };
}

#pragma acc routine seq
inline vec3 operator*(float x, const vec3& rhs) {
  return { x * rhs.x, x * rhs.y, x * rhs.z };
}


/**
 * @struct vec4
 * Structure that mimics CUDA vec4
 */
struct vec4 {
  float x, y, z, w;

  vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}

  vec4(float value) : x(value), y(value), z(value), w(value) {}

  vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}  

  #pragma acc routine seq
  vec4& operator=(const vec3& other) {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  #pragma acc routine seq
  vec4& operator+=(const vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  #pragma acc routine seq
  vec4 operator-(const vec4& rhs) const {
    return { x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w };
  }

  #pragma acc routine seq
  vec4 operator*(const vec4& rhs) const {
    return { x * rhs.x, y * rhs.y, z * rhs.z, w * rhs.w };
  }

  #pragma acc routine seq
  vec3 toVec3() { return vec3(x, y, z); }
};


/**
 * Structure with particle data
 */
struct Particles
{
  const size_t m_Count;
  vec4* m_Positions;
  vec3* m_Velocities;

  Particles(size_t count) : m_Count(count) {
    std::cout << "[Particles] Constructing '" << count << "' particles" << std::endl;
    m_Positions = new vec4[count];
    m_Velocities = new vec3[count];
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(m_Positions[count])
    #pragma acc enter data create(m_Velocities[count])
  }

  ~Particles()
  {
    if (m_Positions) {
      std::cout << "[Particles] Deleting '" << m_Count << "' particles" << std::endl;
      #pragma acc exit data delete(m_Positions)
      #pragma acc exit data delete(m_Velocities)
      #pragma acc exit data delete(this)

      delete[] m_Positions;
      delete[] m_Velocities;
    }
  }

  Particles& operator=(const Particles& src) = delete;

  Particles(const Particles& src) : m_Count(src.m_Count) {
    std::cout << "[Particles] Copying '" << m_Count << "' particles" << std::endl;
    m_Positions = new vec4[m_Count];
    m_Velocities = new vec3[m_Count];
    std::memcpy(m_Positions, src.m_Positions, m_Count * sizeof(vec4));
    std::memcpy(m_Velocities, src.m_Velocities, m_Count * sizeof(vec3));

    #pragma acc enter data copyin(this)
    #pragma acc enter data create(m_Positions[m_Count])
    #pragma acc enter data create(m_Velocities[m_Count])

    #pragma acc parallel loop gang vector_length(1024) present(this, src)
    for (size_t i = 0; i < m_Count; i++) {
      m_Positions[i] = src.m_Positions[i];
      m_Velocities[i] = src.m_Velocities[i];
    }
  }

  /// Would love to use move constructor to avoid unnecessary data copies, however
  /// OpenACC doesn't seem to support move of device pointer or at least I wasn't
  /// able to find a way how to set the device pointer
  
  // Particles(Particles&& src) noexcept : m_Count(src.m_Count), 
  //                                       m_Positions(src.m_Positions),
  //                                       m_Velocities(src.m_Velocities) 
  // {
  //   std::cout << "[Particles] Moving '" << m_Count << "' particles" << std::endl;
  //   #pragma acc enter data copyin(this)
  //   src.m_Positions = nullptr;
  //   src.m_Velocities = nullptr;
  // }

  void updateDevice()
  {
    std::cout << "[Particles] Transfering '" << m_Count << "' particles to GPU" << std::endl;
    #pragma acc update device(m_Positions[0:m_Count])
    #pragma acc update device(m_Velocities[0:m_Count])
  }

  void updateDeviceAsync(unsigned streamID, unsigned waitStreamID)
  {
    #pragma acc update device(m_Positions[m_Count]) async(streamID) wait(waitStreamID)
    #pragma acc update device(m_Velocities[m_Count]) async(streamID) wait(waitStreamID)
  }

  void updateHost()
  {
    std::cout << "[Particles] Transfering '" << m_Count << "' particles to CPU" << std::endl;
    #pragma acc update host(m_Positions[0:m_Count])
    #pragma acc update host(m_Velocities[0:m_Count])
  }

  void updateHostAsync(unsigned streamID, unsigned waitStreamID)
  {
    #pragma acc update host(m_Positions[m_Count]) async(streamID) wait(waitStreamID)
    #pragma acc update host(m_Velocities[m_Count]) async(streamID) wait(waitStreamID)
  }

  void Memset(int value) {
    std::cout << "[Particles] Memset" << std::endl;
    memset(m_Positions, value, m_Count * sizeof(vec4));
    memset(m_Velocities, value, m_Count * sizeof(vec3));
    updateDevice();
  }

  #pragma acc routine seq
  vec4 GetPosition(size_t idx) const {
    return m_Positions[idx];
  }

  #pragma acc routine seq
  void SetPosition(size_t idx, const vec3& value) {
    m_Positions[idx] = value;
  }

  #pragma acc routine seq
  void MovePosition(size_t idx, const vec3& delta) {
    m_Positions[idx] += delta;
  }

  #pragma acc routine seq
  vec3 GetVelocity(size_t idx) const {
    return m_Velocities[idx];
  }

  #pragma acc routine seq
  void SetVelocity(size_t idx, const vec3& value) {
    m_Velocities[idx] = value;
  }
};

/**
 * @struct Velocities
 * Velocities of the particles
 */
struct Velocities
{
  const size_t m_Count;
  vec3* m_Data;

  Velocities(const Velocities& stc) = delete;
  Velocities& operator=(const Velocities& stc) = delete;

  Velocities(size_t count) : m_Count(count)
  {
    std::cout << "Creating velocities" << std::endl;
    m_Data = new vec3[count];

    #pragma acc enter data copyin(this)
    #pragma acc enter data create(m_Data[count])
  }

  ~Velocities()
  {
    std::cout << "Deleting velocities" << std::endl;
    #pragma acc exit data delete(m_Data)
    #pragma acc exit data delete(this)

    delete[] m_Data;
  }

  void updateDevice()
  {
    std::cout << "[Velocities] Updating device" << std::endl;
    #pragma acc update device(m_Data[m_Count])
  }

  void updateHost()
  {
    std::cout << "[Velocities] Updating host" << std::endl;
    #pragma acc update host(m_Data[m_Count])
  }

  void Memset(int value) {
    std::cout << "Memset velocities" << std::endl;
    memset(m_Data, value, m_Count * sizeof(vec3));
    updateDevice();
  }

  void MemsetDeviceMemory(const int value) {
    #pragma acc parallel loop gang vector_length(1024) present(this, m_Data[m_Count])
    for (size_t i = 0; i < m_Count; i++) {
      m_Data[i].x = 0.0f;
      m_Data[i].y = 0.0f;
      m_Data[i].z = 0.0f;
    }
  }

  #pragma acc routine seq
  vec3 GetVelocity(size_t idx) const {
    return m_Data[idx];
  }

  #pragma acc routine seq
  void SetVelocity(size_t idx, const vec3& value) {
    m_Data[idx] = value;
  }

  #pragma acc routine seq
  void AddVelocity(size_t idx, const vec3& value) {
    m_Data[idx] += value;
  }
};

#endif /* __PARTICLES_H__ */
