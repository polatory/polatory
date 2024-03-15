#pragma once

#include <format>
#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/interpolation/rbf_inequality_fitter.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace polatory {

template <int Dim>
class interpolant {
  static constexpr int kDim = Dim;
  using Model = model<kDim>;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Evaluator = interpolation::rbf_evaluator<kDim>;
  using Fitter = interpolation::rbf_fitter<kDim>;
  using IncrementalFitter = interpolation::rbf_incremental_fitter<kDim>;
  using InequalityFitter = interpolation::rbf_inequality_fitter<kDim>;

 public:
  explicit interpolant(const Model& model) : model_(model) {}

  const Points& centers() const {
    throw_if_not_fitted();

    return centers_;
  }

  const Points& grad_centers() const {
    throw_if_not_fitted();

    return centers_;
  }

  common::valuesd evaluate(const Points& points) {
    throw_if_not_fitted();

    set_evaluation_bbox_impl(Bbox::from_points(points));
    return evaluate_impl(points);
  }

  common::valuesd evaluate_impl(const Points& points) const {
    throw_if_not_fitted();

    return evaluator_->evaluate(points);
  }

  void fit(const Points& points, const common::valuesd& values, double absolute_tolerance,
           int max_iter = 100) {
    fit(points, Points(0, kDim), values, absolute_tolerance, absolute_tolerance, max_iter);
  }

  void fit(const Points& points, const Points& grad_points, const common::valuesd& values,
           double absolute_tolerance, double grad_absolute_tolerance, int max_iter = 100) {
    check_num_points(points, grad_points);

    auto n_rhs = points.rows() + kDim * grad_points.rows();
    if (values.rows() != n_rhs) {
      throw std::invalid_argument(std::format("values.rows() must be equal to {}.", n_rhs));
    }

    if (absolute_tolerance <= 0.0) {
      throw std::invalid_argument("absolute_tolerance must be greater than 0.0.");
    }

    if (grad_absolute_tolerance <= 0.0) {
      throw std::invalid_argument("grad_absolute_tolerance must be greater than 0.0.");
    }

    clear();

    Fitter fitter(model_, points, grad_points);
    weights_ = fitter.fit(values, absolute_tolerance, grad_absolute_tolerance, max_iter);

    fitted_ = true;
    centers_ = points;
    grad_centers_ = grad_points;
    bbox_ = Bbox::from_points(centers_).convex_hull(Bbox::from_points(grad_points));
  }

  void fit_incrementally(const Points& points, const common::valuesd& values,
                         double absolute_tolerance, int max_iter = 100) {
    fit_incrementally(points, Points(0, kDim), values, absolute_tolerance, absolute_tolerance,
                      max_iter);
  }

  void fit_incrementally(const Points& points, const Points& grad_points,
                         const common::valuesd& values, double absolute_tolerance,
                         double grad_absolute_tolerance, int max_iter = 100) {
    check_num_points(points, grad_points);

    if (values.rows() != points.rows() + kDim * grad_points.rows()) {
      throw std::invalid_argument(std::format("values.rows() must be equal to {}.",
                                              points.rows() + kDim * grad_points.rows()));
    }

    if (absolute_tolerance <= 0.0) {
      throw std::invalid_argument("absolute_tolerance must be greater than 0.0.");
    }

    if (grad_absolute_tolerance <= 0.0) {
      throw std::invalid_argument("grad_absolute_tolerance must be greater than 0.0.");
    }

    clear();

    IncrementalFitter fitter(model_, points, grad_points);
    std::vector<index_t> center_indices;
    std::vector<index_t> grad_center_indices;
    std::tie(center_indices, grad_center_indices, weights_) =
        fitter.fit(values, absolute_tolerance, grad_absolute_tolerance, max_iter);

    fitted_ = true;
    centers_ = points(center_indices, Eigen::all);
    grad_centers_ = grad_points(grad_center_indices, Eigen::all);
    bbox_ = Bbox::from_points(centers_).convex_hull(Bbox::from_points(grad_centers_));
  }

  void fit_inequality(const Points& points, const common::valuesd& values,
                      const common::valuesd& values_lb, const common::valuesd& values_ub,
                      double absolute_tolerance, int max_iter = 100) {
    if (model_.nugget() > 0.0) {
      throw std::runtime_error("Non-zero nugget is not supported.");
    }

    auto min_n_points = model_.poly_basis_size();
    if (points.rows() < min_n_points) {
      throw std::invalid_argument(
          std::format("points.rows() must be greater than or equal to {}.", min_n_points));
    }

    if (values.rows() != points.rows()) {
      throw std::invalid_argument("values.rows() must be equal to points.rows().");
    }

    if (values_lb.rows() != points.rows()) {
      throw std::invalid_argument("values_lb.rows() must be equal to points.rows().");
    }

    if (values_ub.rows() != points.rows()) {
      throw std::invalid_argument("values_ub.rows() must be equal to points.rows().");
    }

    if (absolute_tolerance <= 0.0) {
      throw std::invalid_argument("absolute_tolerance must be greater than 0.0.");
    }

    clear();

    InequalityFitter fitter(model_, points);
    std::vector<index_t> center_indices;
    std::tie(center_indices, weights_) =
        fitter.fit(values, values_lb, values_ub, absolute_tolerance, max_iter);

    fitted_ = true;
    centers_ = points(center_indices, Eigen::all);
    bbox_ = Bbox::from_points(centers_);
  }

  void set_evaluation_bbox_impl(const Bbox& bbox) {
    throw_if_not_fitted();

    auto union_bbox = bbox.convex_hull(bbox_);

    evaluator_ = std::make_unique<Evaluator>(model_, centers_, grad_centers_, union_bbox,
                                             precision::kPrecise);
    evaluator_->set_weights(weights_);
  }

  const common::valuesd& weights() const {
    throw_if_not_fitted();

    return weights_;
  }

 private:
  void check_num_points(const Points& points, const Points& grad_points) const {
    auto l = model_.poly_basis_size();
    auto mu = points.rows();
    auto sigma = grad_points.rows();

    if (model_.poly_degree() == 1 && mu == 1 && sigma >= 1) {
      // The special case.
      return;
    }

    if (mu < l) {
      throw std::invalid_argument(
          std::format("points.rows() must be greater than or equal to {}.", l));
    }
  }

  void clear() {
    fitted_ = false;
    centers_ = Points();
    grad_centers_ = Points();
    bbox_ = Bbox();
    weights_ = common::valuesd();
  }

  void throw_if_not_fitted() const {
    if (!fitted_) {
      throw std::runtime_error("The interpolant is not fitted yet.");
    }
  }

  const Model model_;

  bool fitted_{};
  Points centers_;
  Points grad_centers_;
  Bbox bbox_;
  common::valuesd weights_;

  std::unique_ptr<Evaluator> evaluator_;
};

}  // namespace polatory
