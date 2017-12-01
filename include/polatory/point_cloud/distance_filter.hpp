// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/exception.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace point_cloud {

class distance_filter {
public:
  distance_filter(const geometry::points3d& points, double distance);

  template <class Derived>
  auto filtered(const Eigen::MatrixBase<Derived>& m) {
    if (m.rows() != n_points_)
      throw common::invalid_argument("m.rows() == " + std::to_string(n_points_));

    return common::take_rows(m, filtered_indices_);
  }

  template <class Derived, class ...Args>
  auto filtered(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
    return std::make_tuple(filtered(m), filtered(std::forward<Args>(args))...);
  }

private:
  const size_t n_points_;

  std::vector<size_t> filtered_indices_;
};

} // namespace point_cloud
} // namespace polatory
