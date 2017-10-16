// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "polatory/polynomial/lagrange_basis.hpp"
#include "polatory/rbf/rbf_base.hpp"

namespace polatory {
namespace preconditioner {

template <class Floating>
class fine_grid2 {
  using Vector3F = Eigen::Matrix<Floating, 3, 1>;
  using VectorXF = Eigen::Matrix<Floating, Eigen::Dynamic, 1>;
  using MatrixXF = Eigen::Matrix<Floating, Eigen::Dynamic, Eigen::Dynamic>;

  const std::vector<size_t> point_idcs_;
  const std::vector<bool> inner_point_;
  const std::vector<size_t> poly_inner_point_idcs_;

public:
  fine_grid2(const rbf::rbf_base& rbf,
             const polynomial::lagrange_basis& poly_basis,
             const std::vector<size_t>& point_indices,
             const std::vector<bool>& inner_point,
             const std::vector<size_t>& poly_point_indices)
    : point_idcs_(point_indices)
    , inner_point_(inner_point)
    , poly_inner_point_idcs_(poly_point_indices) {
  }

  void clear() {

  }

  void setup(const std::vector<Eigen::Vector3d>& points_full) {

  }


};

} // namespace preconditioner
} // namespace polatory
