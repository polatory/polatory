// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>
#include <cmath>

#include <Eigen/Core>

#include "../common/exception.hpp"

namespace polatory {
namespace geometry {

namespace detail {

static Eigen::Matrix4d rotation_matrix(double angle, int axis)
{
   auto c = std::cos(angle);
   auto s = std::sin(angle);

   auto m = Eigen::Matrix4d();
   switch (axis) {
   case 0:
      m <<
         1, 0, 0, 0,
         0, c, -s, 0,
         0, s, c, 0,
         0, 0, 0, 1;
      break;
   case 1:
      m <<
         c, 0, s, 0,
         0, 1, 0, 0,
         -s, 0, c, 0,
         0, 0, 0, 1;
      break;
   case 2:
      m <<
         c, -s, 0, 0,
         s, c, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;
      break;
   default:
      throw common::invalid_parameter("0 <= axis <= 2");
   }

   return m;
}

} // namespace detail

static Eigen::Vector3d affine_transform_point(const Eigen::Vector3d& p, const Eigen::Matrix4d& m)
{
   Eigen::Vector4d p_homo;
   p_homo << p[0], p[1], p[2], 1.0;
   return m.topLeftCorner(3, 4) * p_homo;
}

static Eigen::Vector3d affine_transform_vector(const Eigen::Vector3d& v, const Eigen::Matrix4d& m)
{
   return m.topLeftCorner(3, 3) * v;
}

static Eigen::Matrix4d roll_pitch_yaw_matrix(Eigen::Vector3d angles, std::array<int, 3> axes = { 2, 1, 0 })
{
   return
      detail::rotation_matrix(angles[2], axes[2]) *
      detail::rotation_matrix(angles[1], axes[1]) *
      detail::rotation_matrix(angles[0], axes[0]);
}

static Eigen::Matrix4d scaling_matrix(Eigen::Vector3d scales)
{
   Eigen::Matrix4d m = Eigen::Matrix4d::Zero();

   m.diagonal() << scales[0], scales[1], scales[2], 1.0;

   return m;
}

} // namespace geometry
} // namespace polatory
