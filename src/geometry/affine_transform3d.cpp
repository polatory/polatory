// Copyright (c) 2016, GSI and The Polatory Authors.

#include "polatory/geometry/affine_transform3d.hpp"

#include <cassert>
#include <cmath>

#include <Eigen/Core>

namespace polatory {
namespace geometry {

affine_transform3d::affine_transform3d()
  : m_(Eigen::Matrix4d::Identity()) {
}

affine_transform3d::affine_transform3d(const Eigen::Matrix4d& m)
  : m_(m) {
  assert(m(3, 0) == 0.0);
  assert(m(3, 1) == 0.0);
  assert(m(3, 2) == 0.0);
  assert(m(3, 3) == 1.0);
}

const Eigen::Matrix4d& affine_transform3d::matrix() const {
  return m_;
}

bool affine_transform3d::is_identity() const {
  return m_ == Eigen::Matrix4d::Identity();
}

Eigen::Vector3d affine_transform3d::transform_point(const Eigen::Vector3d& p) const {
  Eigen::Vector4d p_homo;
  p_homo << p(0), p(1), p(2), 1.0;
  return m_.topLeftCorner(3, 4) * p_homo;
}

Eigen::Vector3d affine_transform3d::transform_vector(const Eigen::Vector3d& v) const {
  return m_.topLeftCorner(3, 3) * v;
}

affine_transform3d affine_transform3d::operator*(const affine_transform3d& rhs) const {
  return affine_transform3d(m_ * rhs.m_);
}

affine_transform3d affine_transform3d::roll_pitch_yaw(const Eigen::Vector3d& angles, const std::array<int, 3>& axes) {
  return affine_transform3d(
    rotation_matrix(angles(2), axes[2]) *
    rotation_matrix(angles(1), axes[1]) *
    rotation_matrix(angles(0), axes[0]));
}

affine_transform3d affine_transform3d::scaling(const Eigen::Vector3d& scales) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Zero();

  m.diagonal() << scales(0), scales(1), scales(2), 1.0;

  return affine_transform3d(m);
}

affine_transform3d affine_transform3d::translation(const Eigen::Vector3d& shifts) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Zero();

  m.col(3) << shifts(0), shifts(1), shifts(2), 1.0;

  return affine_transform3d(m);
}

Eigen::Matrix4d affine_transform3d::rotation_matrix(double angle, int axis) {
  assert(axis >= 0 && axis <= 2);

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
    assert(false);
  }

  return m;
}

} // namespace geometry
} // namespace polatory
