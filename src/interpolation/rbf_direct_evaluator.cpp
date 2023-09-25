#include <polatory/interpolation/rbf_direct_evaluator.hpp>
#include <polatory/numeric/sum_accumulator.hpp>

namespace polatory::interpolation {

rbf_direct_evaluator::rbf_direct_evaluator(const model& model,
                                           const geometry::points3d& source_points)
    : rbf_direct_evaluator(model, source_points, geometry::points3d(0, 3)) {}

rbf_direct_evaluator::rbf_direct_evaluator(const model& model,
                                           const geometry::points3d& source_points,
                                           const geometry::points3d& source_grad_points)
    : model_(model),
      dim_(model.poly_dimension()),
      l_(model.poly_basis_size()),
      mu_(source_points.rows()),
      sigma_(source_grad_points.rows()),
      src_points_(source_points),
      src_grad_points_(source_grad_points) {
  if (l_ > 0) {
    p_ = std::make_unique<PolynomialEvaluator>(model.poly_dimension(), model.poly_degree());
  }
}

// TODO: Use Kahan summation.
common::valuesd rbf_direct_evaluator::evaluate() const {
  common::valuesd y(fld_mu_ + dim_ * fld_sigma_);

  auto w = weights_.head(mu_);
  auto grad_w = weights_.segment(mu_, dim_ * sigma_).reshaped<Eigen::RowMajor>(sigma_, dim_);

  const auto& rbf = model_.rbf();
  for (index_t i = 0; i < fld_mu_; i++) {
    for (index_t j = 0; j < mu_; j++) {
      y(i) += w(j) * rbf.evaluate(fld_points_.row(i) - src_points_.row(j));
    }

    for (index_t j = 0; j < sigma_; j++) {
      y(i) += grad_w.row(j).dot(
          -rbf.evaluate_gradient(fld_points_.row(i) - src_grad_points_.row(j)).head(dim_));
    }
  }

  for (index_t i = 0; i < fld_sigma_; i++) {
    for (index_t j = 0; j < mu_; j++) {
      y.segment(fld_mu_ + dim_ * i, dim_) +=
          w(j) * -rbf.evaluate_gradient(fld_grad_points_.row(i) - src_points_.row(j))
                      .head(dim_)
                      .transpose();
    }

    for (index_t j = 0; j < sigma_; j++) {
      y.segment(fld_mu_ + dim_ * i, dim_) +=
          (grad_w.row(j) * rbf.evaluate_hessian(fld_grad_points_.row(i) - src_grad_points_.row(j))
                               .topLeftCorner(dim_, dim_))
              .transpose();
    }
  }

  if (l_ > 0) {
    // Add polynomial terms.
    y += p_->evaluate();
  }

  return y;
}

void rbf_direct_evaluator::set_field_points(const geometry::points3d& field_points) {
  set_field_points(field_points, geometry::points3d(0, 3));
}

void rbf_direct_evaluator::set_field_points(const geometry::points3d& field_points,
                                            const geometry::points3d& field_grad_points) {
  fld_mu_ = static_cast<index_t>(field_points.rows());
  fld_sigma_ = static_cast<index_t>(field_grad_points.rows());

  fld_points_ = field_points;
  fld_grad_points_ = field_grad_points;

  if (l_ > 0) {
    p_->set_field_points(fld_points_, fld_grad_points_);
  }
}

}  // namespace polatory::interpolation
