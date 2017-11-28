// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <random>
#include <set>
#include <vector>

#include <polatory/common/bsearch.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/basis_base.hpp>

namespace polatory {
namespace polynomial {

class unisolvent_point_set {
public:
  unisolvent_point_set(const geometry::vectors3d& points,
                       const std::vector<size_t>& point_indices,
                       int dimension,
                       int degree)
    : n_points_(points.rows())
    , n_poly_basis_(polynomial::basis_base::basis_size(dimension, degree))
    , point_idcs_(point_indices) {
    if (degree < 0) return;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, point_indices.size() - 1);
    std::set<size_t> set;

    while (set.size() < n_poly_basis_) {
      size_t point_idx = point_idcs_[dist(gen)];
      if (!set.insert(point_idx).second)
        continue;

      auto it = common::bsearch_eq(point_idcs_.begin(), point_idcs_.end(), point_idx);
      point_idcs_.erase(it);
    }

    point_idcs_.insert(point_idcs_.begin(), set.begin(), set.end());

    assert(point_idcs_.size() == point_indices.size());
  }

  const std::vector<size_t> point_indices() const {
    return point_idcs_;
  }

private:
  const size_t n_points_;
  const size_t n_poly_basis_;

  std::vector<size_t> point_idcs_;
};

} // namespace polynomial
} // namespace polatory
