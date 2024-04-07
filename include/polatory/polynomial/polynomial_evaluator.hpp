#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>

namespace polatory::polynomial {

template <class Basis>
class polynomial_evaluator {
  static constexpr int kDim = Basis::kDim;
  using Points = geometry::pointsNd<kDim>;

 public:
  explicit polynomial_evaluator(int degree)
      : basis_(degree), weights_(vectord::Zero(basis_.basis_size())) {}

  vectord evaluate() const {
    auto p = basis_.evaluate(points_, grad_points_);

    return p * weights_;
  }

  void set_target_points(const Points& points, const Points& grad_points) {
    points_ = points;
    grad_points_ = grad_points;
  }

  void set_weights(const vectord& weights) {
    POLATORY_ASSERT(weights.rows() == basis_.basis_size());

    weights_ = weights;
  }

 private:
  const Basis basis_;

  Points points_;
  Points grad_points_;
  vectord weights_;
};

}  // namespace polatory::polynomial
