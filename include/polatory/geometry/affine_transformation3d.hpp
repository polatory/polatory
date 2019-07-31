// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>

#include <Eigen/Core>

#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace geometry {

class affine_transformation3d {
public:
  affine_transformation3d();

  explicit affine_transformation3d(const Eigen::Matrix4d& m);

  affine_transformation3d inverse() const;

  bool is_identity() const;

  const Eigen::Matrix4d& matrix() const;

  bool operator==(const affine_transformation3d& rhs) const;

  bool operator!=(const affine_transformation3d& rhs) const;

  affine_transformation3d operator*(const affine_transformation3d& rhs) const;

  point3d transform_point(const point3d& p) const;

  vector3d transform_vector(const vector3d& v) const;

  static affine_transformation3d roll_pitch_yaw(const vector3d& angles, const std::array<int, 3>& axes = { 2, 1, 0 });

  static affine_transformation3d scaling(const vector3d& scales);

  static affine_transformation3d translation(const vector3d& shifts);

private:
  static Eigen::Matrix4d rotation_matrix(double angle, int axis);

  Eigen::Matrix4d m_;
};

}  // namespace geometry
}  // namespace polatory
