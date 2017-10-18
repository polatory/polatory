// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <random>
#include <set>
#include <vector>

#include <Eigen/Core>

#include "polatory/common/bsearch.hpp"
#include "polatory/polynomial/basis_base.hpp"

namespace polatory {
namespace polynomial {

class unisolvent_point_set {
  const size_t n_points;
  const size_t n_polynomials;

  std::vector<size_t> point_idcs_;

public:
  template <typename Container>
  unisolvent_point_set(const Container& points,
                       const std::vector<size_t>& point_indices,
                       int dimension,
                       int degree)
    : n_points(points.size())
    , n_polynomials(polynomial::basis_base::basis_size(dimension, degree))
    , point_idcs_(point_indices) {
    if (degree < 0) return;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, point_indices.size() - 1);
    std::set<size_t> set;

    while (set.size() < n_polynomials) {
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
};

} // namespace polynomial
} // namespace polatory
