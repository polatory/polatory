// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <polatory/common/bsearch.hpp>
#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace polynomial {

class unisolvent_point_set {
  static constexpr int kMaxTrial = 32;

public:
  unisolvent_point_set(const geometry::vectors3d& points,
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
    auto set_is_unisolvent = false;
    auto trial = 0;
    while (!set_is_unisolvent && trial < kMaxTrial) {
      set.clear();

      while (static_cast<index_t>(set.size()) < n_poly_basis) {
        auto point_idx = point_indices[dist(gen)];
        set.insert(point_idx);
      }

      try {
        lagrange_basis basis(dimension, degree, common::take_rows(points, set));
        set_is_unisolvent = true;
      } catch (const std::domain_error&) {
        // noop.
      }
      trial++;
    }

    if (!set_is_unisolvent && trial == kMaxTrial) {
      throw std::runtime_error("Could not find a unisolvent set of points.");
    }

    for (auto idx : set) {
      auto it = common::bsearch_eq(point_idcs_.begin(), point_idcs_.end(), idx);
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
