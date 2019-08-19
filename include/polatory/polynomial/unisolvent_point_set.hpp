// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <random>
#include <set>
#include <vector>

#include <polatory/common/bsearch.hpp>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace polynomial {

class unisolvent_point_set {
public:
  unisolvent_point_set(const geometry::vectors3d& /*points*/,
                       const std::vector<index_t>& point_indices,
                       int dimension,
                       int degree)
    : point_idcs_(point_indices) {
    if (degree < 0) return;

    auto n_points = static_cast<index_t>(point_indices.size());
    auto n_poly_basis = polynomial_basis_base::basis_size(dimension, degree);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dist(index_t{ 0 }, n_points - 1);
    std::set<index_t> set;

    while (static_cast<index_t>(set.size()) < n_poly_basis) {
      auto point_idx = point_idcs_[dist(gen)];
      if (!set.insert(point_idx).second)
        continue;

      auto it = common::bsearch_eq(point_idcs_.begin(), point_idcs_.end(), point_idx);
      point_idcs_.erase(it);
    }

    point_idcs_.insert(point_idcs_.begin(), set.begin(), set.end());

    POLATORY_ASSERT(point_idcs_.size() == point_indices.size());
  }

  const std::vector<index_t>& point_indices() const {
    return point_idcs_;
  }

private:
  std::vector<index_t> point_idcs_;
};

}  // namespace polynomial
}  // namespace polatory
