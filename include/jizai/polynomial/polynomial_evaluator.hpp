#pragma once

#include <jizai/common/macros.hpp>
#include <jizai/geometry/point3d.hpp>
#include <jizai/types.hpp>

namespace jizai::polynomial {

template <class Basis>
class PolynomialEvaluator {
  static constexpr int kDim = Basis::kDim;
  using Points = geometry::Points<kDim>;

 public:
  explicit PolynomialEvaluator(int degree)
      : basis_(degree), weights_(VecX::Zero(basis_.basis_size())) {}

  VecX evaluate() const {
    auto p = basis_.evaluate(points_, grad_points_);

    return p * weights_;
  }

  void set_target_points(const Points& points, const Points& grad_points) {
    points_ = points;
    grad_points_ = grad_points;
  }

  void set_weights(const VecX& weights) {
    JIZAI_ASSERT(weights.rows() == basis_.basis_size());

    weights_ = weights;
  }

 private:
  const Basis basis_;

  Points points_;
  Points grad_points_;
  VecX weights_;
};

}  // namespace jizai::polynomial
