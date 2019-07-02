// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/geometry/affine_transformation3d.hpp>

#include <cassert>
#include <cmath>

#include <polatory/common/exception.hpp>

namespace polatory {
namespace geometry {

affine_transformation3d::affine_transformation3d()
  : m_(Eigen::Matrix4d::Identity()) {
}

affine_transformation3d::affine_transformation3d(const Eigen::Matrix4d& m)
  : m_(m) {
  if (m.row(3) != Eigen::RowVector4d(0, 0, 0, 1))
    throw common::invalid_argument("m.row(3) must be (0, 0, 0, 1).");
}

affine_transformation3d affine_transformation3d::inverse() const {
  Eigen::Matrix3d ai = m_.topLeftCorner<3, 3>().inverse();
  Eigen::Matrix4d mi = Eigen::Matrix4d::Identity();
  mi.topLeftCorner<3, 3>() = ai;
  mi.topRightCorner<3, 1>() = -ai * m_.topRightCorner<3, 1>();
  return affine_transformation3d(mi);
}

bool affine_transformation3d::is_identity() const {
  return m_ == Eigen::Matrix4d::Identity();
}

const Eigen::Matrix4d& affine_transformation3d::matrix() const {
  return m_;
}

point3d affine_transformation3d::transform_point(const point3d& p) const {
  return m_.topLeftCorner(3, 4) * p.homogeneous().transpose();
}

vector3d affine_transformation3d::transform_vector(const vector3d& v) const {
  return m_.topLeftCorner(3, 3) * v.transpose();
}

affine_transformation3d affine_transformation3d::operator*(const affine_transformation3d& rhs) const {
  return affine_transformation3d(m_ * rhs.m_);
}

affine_transformation3d affine_transformation3d::roll_pitch_yaw(const vector3d& angles, const std::array<int, 3>& axes) {
  return affine_transformation3d(
    rotation_matrix(angles(2), axes[2]) *
    rotation_matrix(angles(1), axes[1]) *
    rotation_matrix(angles(0), axes[0]));
}

affine_transformation3d affine_transformation3d::scaling(const vector3d& scales) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Zero();

  m.diagonal() << scales(0), scales(1), scales(2), 1.0;

  return affine_transformation3d(m);
}

affine_transformation3d affine_transformation3d::translation(const vector3d& shifts) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Zero();

  m.col(3) << shifts(0), shifts(1), shifts(2), 1.0;

  return affine_transformation3d(m);
}

Eigen::Matrix4d affine_transformation3d::rotation_matrix(double angle, int axis) {
  assert(axis >= 0 && axis <= 2);

  auto c = std::cos(angle);
  auto s = std::sin(angle);

  Eigen::Matrix4d m;
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

}  // namespace geometry
}  // namespace polatory
