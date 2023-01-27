#pragma once

#include <algorithm>
#include <memory>
#include <polatory/common/eigen_utility.hpp>
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

class interpolant {
 public:
  explicit interpolant(model model) : model_(std::move(model)), fitted_(false) {}

  const geometry::points3d& centers() const {
    if (!fitted_) throw std::runtime_error(kNotFittedErrorMessage);

    return centers_;
  }

  common::valuesd evaluate(const geometry::points3d& points) {
    if (!fitted_) throw std::runtime_error(kNotFittedErrorMessage);

    set_evaluation_bbox_impl(geometry::bbox3d::from_points(points));
    return evaluate_impl(points);
  }

  common::valuesd evaluate_impl(const geometry::points3d& points) const {
    if (!fitted_) throw std::runtime_error(kNotFittedErrorMessage);

    return evaluator_->evaluate(points);
  }

  void fit(const geometry::points3d& points, const common::valuesd& values,
           double absolute_tolerance) {
    auto min_n_points = std::max(1, model_.poly_basis_size());
    if (points.rows() < min_n_points)
      throw std::invalid_argument("points.rows() must be greater than or equal to " +
                                  std::to_string(min_n_points) + ".");

    if (values.rows() != points.rows())
      throw std::invalid_argument("values.rows() must be equal to points.rows().");

    if (absolute_tolerance <= 0.0)
      throw std::invalid_argument("absolute_tolerance must be greater than 0.0.");

    clear();

    interpolation::rbf_fitter fitter(model_, points);
    weights_ = fitter.fit(values, absolute_tolerance);

    fitted_ = true;
    centers_ = points;
    centers_bbox_ = geometry::bbox3d::from_points(centers_);
  }

  void fit_incrementally(const geometry::points3d& points, const common::valuesd& values,
                         double absolute_tolerance) {
    if (model_.nugget() > 0.0) throw std::runtime_error("Non-zero nugget is not supported.");

    auto min_n_points = std::max(1, model_.poly_basis_size());
    if (points.rows() < min_n_points)
      throw std::invalid_argument("points.rows() must be greater than or equal to " +
                                  std::to_string(min_n_points) + ".");

    if (values.rows() != points.rows())
      throw std::invalid_argument("values.rows() must be equal to points.rows().");

    if (absolute_tolerance <= 0.0)
      throw std::invalid_argument("absolute_tolerance must be greater than 0.0.");

    clear();

    interpolation::rbf_incremental_fitter fitter(model_, points);
    std::vector<index_t> center_indices;
    std::tie(center_indices, weights_) = fitter.fit(values, absolute_tolerance);

    fitted_ = true;
    centers_ = common::take_rows(points, center_indices);
    centers_bbox_ = geometry::bbox3d::from_points(centers_);
  }

  void fit_inequality(const geometry::points3d& points, const common::valuesd& values,
                      const common::valuesd& values_lb, const common::valuesd& values_ub,
                      double absolute_tolerance) {
    if (model_.nugget() > 0.0) throw std::runtime_error("Non-zero nugget is not supported.");

    auto min_n_points = std::max(1, model_.poly_basis_size());
    if (points.rows() < min_n_points)
      throw std::invalid_argument("points.rows() must be greater than or equal to " +
                                  std::to_string(min_n_points) + ".");

    if (values.rows() != points.rows())
      throw std::invalid_argument("values.rows() must be equal to points.rows().");

    if (values_lb.rows() != points.rows())
      throw std::invalid_argument("values_lb.rows() must be equal to points.rows().");

    if (values_ub.rows() != points.rows())
      throw std::invalid_argument("values_ub.rows() must be equal to points.rows().");

    if (absolute_tolerance <= 0.0)
      throw std::invalid_argument("absolute_tolerance must be greater than 0.0.");

    clear();

    interpolation::rbf_inequality_fitter fitter(model_, points);
    std::vector<index_t> center_indices;
    std::tie(center_indices, weights_) =
        fitter.fit(values, values_lb, values_ub, absolute_tolerance);

    fitted_ = true;
    centers_ = common::take_rows(points, center_indices);
    centers_bbox_ = geometry::bbox3d::from_points(centers_);
  }

  void set_evaluation_bbox_impl(const geometry::bbox3d& bbox) {
    if (!fitted_) throw std::runtime_error(kNotFittedErrorMessage);

    auto union_bbox = bbox.union_hull(centers_bbox_);

    evaluator_ = std::make_unique<interpolation::rbf_evaluator<>>(model_, centers_, union_bbox);
    evaluator_->set_weights(weights_);
  }

  const common::valuesd& weights() const {
    if (!fitted_) throw std::runtime_error(kNotFittedErrorMessage);

    return weights_;
  }

 private:
  void clear() {
    fitted_ = false;
    centers_ = geometry::points3d();
    centers_bbox_ = geometry::bbox3d();
    weights_ = common::valuesd();
  }

  static constexpr const char* kNotFittedErrorMessage = "The interpolant is not fitted yet.";

  const model model_;

  bool fitted_;
  geometry::points3d centers_;
  geometry::bbox3d centers_bbox_;
  common::valuesd weights_;

  std::unique_ptr<interpolation::rbf_evaluator<>> evaluator_;
};

}  // namespace polatory
