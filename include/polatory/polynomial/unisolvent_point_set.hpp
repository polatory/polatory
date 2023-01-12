#pragma once

#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/types.hpp>

namespace polatory {
namespace polynomial {

// TODO(mizuno): If given points have a large offset from the origin,
// construction can fail due to a large condition number.
class unisolvent_point_set {
  static constexpr int kMaxTrial = 32;

public:
  unisolvent_point_set(const geometry::vectors3d& points,
                       int dimension,
                       int degree) {
    if (degree < 0) return;

    auto n_points = static_cast<index_t>(points.rows());
    auto n_poly_basis = polynomial_basis_base::basis_size(dimension, degree);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dist(index_t{ 0 }, n_points - 1);

    std::set<index_t> set;
    auto found = false;
    auto trial = 0;
    while (!found && trial < kMaxTrial) {
      set.clear();

      while (static_cast<index_t>(set.size()) < n_poly_basis) {
        set.insert(dist(gen));
      }

      try {
        lagrange_basis basis(dimension, degree, common::take_rows(points, set));
        found = true;
      } catch (const std::domain_error&) {
        // noop.
      }

      trial++;
    }

    if (!found && trial == kMaxTrial) {
      throw std::runtime_error("Could not find a unisolvent set of points.");
    }

    point_idcs_.insert(point_idcs_.begin(), set.begin(), set.end());
  }

  const std::vector<index_t>& point_indices() const {
    return point_idcs_;
  }

private:
  std::vector<index_t> point_idcs_;
};

}  // namespace polynomial
}  // namespace polatory
