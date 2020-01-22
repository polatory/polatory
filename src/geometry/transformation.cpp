// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/geometry/transformation.hpp>

#include <cmath>
#include <stdexcept>

#include <polatory/common/macros.hpp>

namespace polatory {
namespace geometry {

Eigen::Matrix3d scaling(const vector3d& scales) {
  Eigen::Matrix3d m = Eigen::Matrix3d::Identity();

  m.diagonal() << scales(0), scales(1), scales(2);

  return m;
}

Eigen::Matrix3d rotation(double angle, int axis) {
  if (axis < 0 || axis > 2)
    throw std::invalid_argument("axis must be within the range of 0 to 2.");

  auto c = std::cos(angle);
  auto s = std::sin(angle);

  Eigen::Matrix3d m;
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
    POLATORY_NEVER_REACH();
    break;
  }

  return m;
}

}  // namespace geometry
}  // namespace polatory
