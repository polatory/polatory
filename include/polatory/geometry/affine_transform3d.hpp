// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <array>

#include <Eigen/Core>

namespace polatory {
namespace geometry {

class affine_transform3d {
public:
  affine_transform3d();

  explicit affine_transform3d(const Eigen::Matrix4d& m);

  const Eigen::Matrix4d& matrix() const;

  Eigen::Vector3d transform_point(const Eigen::Vector3d& p) const;

  Eigen::Vector3d transform_vector(const Eigen::Vector3d& v) const;

  affine_transform3d operator*(const affine_transform3d& rhs) const;

  static affine_transform3d roll_pitch_yaw(const Eigen::Vector3d& angles, const std::array<int, 3>& axes = { 2, 1, 0 });

  static affine_transform3d scaling(const Eigen::Vector3d& scales);

  static affine_transform3d translation(const Eigen::Vector3d& shifts);

private:
  static Eigen::Matrix4d rotation_matrix(double angle, int axis);

  Eigen::Matrix4d m_;
};

} // namespace geometry
} // namespace polatory