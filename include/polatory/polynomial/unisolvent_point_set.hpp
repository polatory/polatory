#pragma once

#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/types.hpp>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace polatory::polynomial {

template <int Dim>
class unisolvent_point_set {
  static constexpr int kMaxTrial = 32;

  using Points = geometry::pointsNd<Dim>;

 public:
  unisolvent_point_set(const Points& points, int degree) {
    if (degree < 0) {
      return;
    }

    auto n_points = points.rows();
    auto n_poly_basis = polynomial_basis_base<Dim>::basis_size(degree);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dist(index_t{0}, n_points - 1);

    std::set<index_t> best_set;
    auto best_rcond = 0.0;
    auto found = false;

    for (auto trial = 0; trial < kMaxTrial; trial++) {
      std::set<index_t> set;

      while (static_cast<index_t>(set.size()) < n_poly_basis) {
        set.insert(dist(gen));
      }

      try {
        lagrange_basis<Dim> basis(degree,
                                  points(std::vector<index_t>(set.begin(), set.end()), Eigen::all));

        if (best_rcond < basis.rcond()) {
          best_rcond = basis.rcond();
          best_set = std::move(set);
        }

        found = true;
      } catch (const std::domain_error&) {
        // No-op.
      }
    }

    if (!found) {
      throw std::runtime_error("Could not find a unisolvent set of points.");
    }

    point_idcs_.insert(point_idcs_.begin(), best_set.begin(), best_set.end());
  }

  // Returns the sorted point indices.
  const std::vector<index_t>& point_indices() const { return point_idcs_; }

 private:
  std::vector<index_t> point_idcs_;
};

}  // namespace polatory::polynomial
