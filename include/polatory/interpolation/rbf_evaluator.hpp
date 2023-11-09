#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_evaluator.hpp>
#include <polatory/precision.hpp>
#include <polatory/types.hpp>

namespace polatory::interpolation {

template <class Model>
class rbf_evaluator {
  static constexpr int kDim = Model::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using MonomialBasis = polynomial::monomial_basis<kDim>;
  using PolynomialEvaluator = polynomial::polynomial_evaluator<MonomialBasis>;

 public:
  rbf_evaluator(const Model& model, const Points& source_points, precision prec)
      : rbf_evaluator(model, source_points, Points(0, kDim), prec) {}

  rbf_evaluator(const Model& model, const Points& source_points, const Points& source_grad_points,
                precision prec)
      : rbf_evaluator(
            model, source_points, source_grad_points,
            Bbox::from_points(source_points).convex_hull(Bbox::from_points(source_grad_points)),
            prec) {}

  rbf_evaluator(const Model& model, const Points& source_points, const Bbox& bbox, precision prec)
      : rbf_evaluator(model, source_points, Points(0, kDim), bbox, prec) {}

  rbf_evaluator(const Model& model, const Points& source_points, const Points& source_grad_points,
                const Bbox& bbox, precision prec)
      : rbf_evaluator(model, bbox, prec) {
    set_source_points(source_points, source_grad_points);
  }

  rbf_evaluator(const Model& model, const Bbox& bbox, precision prec)
      : l_(model.poly_basis_size()),
        a_(model, bbox, prec),
        f_(model, bbox, prec),
        ft_(model, bbox, prec),
        h_(model, bbox, prec) {
    if (l_ > 0) {
      p_ = std::make_unique<PolynomialEvaluator>(model.poly_degree());
    }
  }

  common::valuesd evaluate() const {
    common::valuesd y = common::valuesd::Zero(trg_mu_ + kDim * trg_sigma_);

    y.head(trg_mu_) += a_.evaluate();
    y.head(trg_mu_) += f_.evaluate();
    y.tail(kDim * trg_sigma_) += ft_.evaluate();
    y.tail(kDim * trg_sigma_) += h_.evaluate();

    if (l_ > 0) {
      // Add polynomial terms.
      y += p_->evaluate();
    }

    return y;
  }

  common::valuesd evaluate(const Points& target_points) {
    return evaluate(target_points, Points(0, kDim));
  }

  common::valuesd evaluate(const Points& target_points, const Points& target_grad_points) {
    set_target_points(target_points, target_grad_points);

    return evaluate();
  }

  void set_source_points(const Points& points, const Points& grad_points) {
    mu_ = points.rows();
    sigma_ = grad_points.rows();

    a_.set_source_points(points);
    f_.set_source_points(grad_points);
    ft_.set_source_points(points);
    h_.set_source_points(grad_points);
  }

  void set_target_points(const Points& points) { set_target_points(points, Points(0, kDim)); }

  void set_target_points(const Points& points, const Points& grad_points) {
    trg_mu_ = points.rows();
    trg_sigma_ = grad_points.rows();

    a_.set_target_points(points);
    f_.set_target_points(points);
    ft_.set_target_points(grad_points);
    h_.set_target_points(grad_points);

    if (l_ > 0) {
      p_->set_target_points(points, grad_points);
    }
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    POLATORY_ASSERT(weights.rows() == mu_ + kDim * sigma_ + l_);

    a_.set_weights(weights.head(mu_));
    f_.set_weights(weights.segment(mu_, kDim * sigma_));
    ft_.set_weights(weights.head(mu_));
    h_.set_weights(weights.segment(mu_, kDim * sigma_));

    if (l_ > 0) {
      p_->set_weights(weights.tail(l_));
    }
  }

 private:
  const index_t l_;
  index_t mu_{};
  index_t sigma_{};
  index_t trg_mu_{};
  index_t trg_sigma_{};

  fmm::fmm_evaluator<Model> a_;
  fmm::fmm_gradient_evaluator<Model> f_;
  fmm::fmm_gradient_transpose_evaluator<Model> ft_;
  fmm::fmm_hessian_evaluator<Model> h_;
  std::unique_ptr<PolynomialEvaluator> p_;
};

}  // namespace polatory::interpolation
