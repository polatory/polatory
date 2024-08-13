#pragma once

#include <Eigen/Core>
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
class UnisolventPointSet {
  static constexpr int kDim = Dim;
  using LagrangeBasis = LagrangeBasis<kDim>;
  using Points = geometry::Points<kDim>;
  using PolynomialBasisBase = PolynomialBasisBase<kDim>;

  static constexpr int kNumTrials = 100;

 public:
  UnisolventPointSet(const Points& points, int degree) {
    if (degree < 0) {
      return;
    }

    auto n_points = points.rows();
    auto n_poly_basis = PolynomialBasisBase::basis_size(degree);

    std::mt19937 gen;
    std::uniform_int_distribution<Index> dist(Index{0}, n_points - 1);

    std::set<Index> best_set;
    auto best_rcond = 0.0;
    auto found = false;

    for (auto trial = 0; trial < kNumTrials; trial++) {
      std::set<Index> set;

      while (static_cast<Index>(set.size()) < n_poly_basis) {
        set.insert(dist(gen));
      }

      try {
        LagrangeBasis basis(degree, points(std::vector<Index>(set.begin(), set.end()), Eigen::all));

        if (best_rcond < basis.rcond()) {
          best_rcond = basis.rcond();
          best_set = std::move(set);
        }

        found = true;
      } catch (const std::invalid_argument&) {  // NOLINT(bugprone-empty-catch)
        // No-op.
      }
    }

    if (!found) {
      throw std::runtime_error("could not find a unisolvent set of points");
    }

    point_idcs_.insert(point_idcs_.begin(), best_set.begin(), best_set.end());
  }

  // Returns the *sorted* point indices.
  const std::vector<Index>& point_indices() const { return point_idcs_; }

 private:
  std::vector<Index> point_idcs_;
};

}  // namespace polatory::polynomial
