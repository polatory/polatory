#pragma once

#include <array>
#include <cmath>

#include <Eigen/Core>

#include "common/exception.hpp"

namespace polatory {
namespace geometry {

namespace detail {

static Eigen::Matrix3d rotation_matrix(double angle, int axis)
{
   auto c = std::cos(angle);
   auto s = std::sin(angle);

   auto m = Eigen::Matrix3d();
   switch (axis) {
   case 0:
      m <<
         1, 0, 0,
         0, c, -s,
         0, s, c;
      break;
   case 1:
      m <<
         c, 0, s,
         0, 1, 0,
         -s, 0, c;
      break;
   case 2:
      m <<
         c, -s, 0,
         s, c, 0,
         0, 0, 1;
      break;
   default:
      throw common::invalid_parameter("0 <= axis <= 2");
   }

   return m;
}

} // namespace detail

static Eigen::Matrix3d roll_pitch_yaw_matrix(Eigen::Vector3d angles, std::array<int, 3> axes = { 2, 1, 0 })
{
   return
      detail::rotation_matrix(angles[2], axes[2]) *
      detail::rotation_matrix(angles[1], axes[1]) *
      detail::rotation_matrix(angles[0], axes[0]);
}

static Eigen::Matrix3d scaling_matrix(Eigen::Vector3d scales)
{
   return scales.asDiagonal();
}

} // namespace geometry
} // namespace polatory
