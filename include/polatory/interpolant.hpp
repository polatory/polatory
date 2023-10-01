#pragma once

#include <algorithm>
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

template <class Model>
class interpolant {
  static constexpr int kDim = Model::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Evaluator = interpolation::rbf_evaluator<Model>;

 public:
  explicit interpolant(const Model& model) : model_(std::move(model)) {}

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
           int max_iter = 32) {
    fit(points, Points(0, kDim), values, absolute_tolerance, absolute_tolerance, max_iter);
  }

  void fit(const Points& points, const Points& grad_points, const common::valuesd& values,
           double absolute_tolerance, double grad_absolute_tolerance, int max_iter = 32) {
    auto min_n_points = std::max(index_t{1}, model_.poly_basis_size());
    if (points.rows() < min_n_points) {
      throw std::invalid_argument(std::format("points.rows() must be greater than or equal to {}.",
                                              std::to_string(min_n_points)));
    }

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

    interpolation::rbf_fitter fitter(model_, points, grad_points);
    weights_ = fitter.fit(values, absolute_tolerance, grad_absolute_tolerance, max_iter);

    fitted_ = true;
    centers_ = points;
    grad_centers_ = grad_points;
    bbox_ = Bbox::from_points(centers_).convex_hull(Bbox::from_points(grad_points));
  }

  void fit_incrementally(const Points& points, const common::valuesd& values,
                         double absolute_tolerance, int max_iter = 32) {
    auto min_n_points = std::max(index_t{1}, model_.poly_basis_size());
    if (points.rows() < min_n_points) {
      throw std::invalid_argument("points.rows() must be greater than or equal to " +
                                  std::to_string(min_n_points) + ".");
    }

    if (values.rows() != points.rows()) {
      throw std::invalid_argument("values.rows() must be equal to points.rows().");
    }

    if (absolute_tolerance <= 0.0) {
      throw std::invalid_argument("absolute_tolerance must be greater than 0.0.");
    }

    clear();

    interpolation::rbf_incremental_fitter fitter(model_, points);
    std::vector<index_t> center_indices;
    std::tie(center_indices, weights_) = fitter.fit(values, absolute_tolerance, max_iter);

    fitted_ = true;
    centers_ = points(center_indices, Eigen::all);
    bbox_ = Bbox::from_points(centers_);
  }

  void fit_inequality(const Points& points, const common::valuesd& values,
                      const common::valuesd& values_lb, const common::valuesd& values_ub,
                      double absolute_tolerance, int max_iter = 32) {
    if (model_.nugget() > 0.0) {
      throw std::runtime_error("Non-zero nugget is not supported.");
    }

    auto min_n_points = std::max(index_t{1}, model_.poly_basis_size());
    if (points.rows() < min_n_points) {
      throw std::invalid_argument("points.rows() must be greater than or equal to " +
                                  std::to_string(min_n_points) + ".");
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

    interpolation::rbf_inequality_fitter fitter(model_, points);
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
