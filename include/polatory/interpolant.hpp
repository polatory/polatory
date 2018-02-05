// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <polatory/common/types.hpp>
#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/exception.hpp>
#include <polatory/geometry/affine_transform3d.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/interpolation/rbf_inequality_fitter.hpp>
#include <polatory/model.hpp>

namespace polatory {

class interpolant {
public:
  explicit interpolant(const model& model)
    : model_(model) {
  }

  const geometry::points3d& centers() const {
    return centers_;
  }

  geometry::bbox3d centers_bbox() const {
    return centers_bbox_;
  }

  common::valuesd evaluate_points(const geometry::points3d& points) {
    set_evaluation_bbox_impl(geometry::bbox3d::from_points(points));

    return evaluate_points_impl(points);
  }

  common::valuesd evaluate_points_impl(const geometry::points3d& points) const {
    auto transformed = affine_transform_points(points);

    return evaluator_->evaluate_points(transformed);
  }

  void fit(const geometry::points3d& points, const common::valuesd& values,
           double absolute_tolerance) {
    auto min_n_points = model_.poly_basis_size() + 1;
    if (points.rows() < min_n_points)
      throw common::invalid_argument("points.rows() >= " + std::to_string(min_n_points));

    if (values.rows() != points.rows())
      throw common::invalid_argument("values.rows() == points.rows()");

    if (absolute_tolerance <= 0.0)
      throw common::invalid_argument("absolute_tolerance > 0.0");

    clear_centers();

    auto transformed = affine_transform_points(points);
    interpolation::rbf_fitter fitter(model_, transformed);

    centers_ = std::move(transformed);
    centers_bbox_ = geometry::bbox3d::from_points(centers_);
    weights_ = fitter.fit(values, absolute_tolerance);
  }

  void fit_incrementally(const geometry::points3d& points, const common::valuesd& values,
                         double absolute_tolerance) {
    if (model_.rbf().nugget() > 0.0)
      throw common::not_supported("RBF with finite nugget");

    auto min_n_points = model_.poly_basis_size() + 1;
    if (points.rows() < min_n_points)
      throw common::invalid_argument("points.rows() >= " + std::to_string(min_n_points));

    if (values.rows() != points.rows())
      throw common::invalid_argument("values.rows() == points.rows()");

    if (absolute_tolerance <= 0.0)
      throw common::invalid_argument("absolute_tolerance > 0.0");

    clear_centers();

    auto transformed = affine_transform_points(points);
    interpolation::rbf_incremental_fitter fitter(model_, transformed);

    std::vector<size_t> center_indices;
    std::tie(center_indices, weights_) = fitter.fit(values, absolute_tolerance);

    centers_ = common::take_rows(transformed, center_indices);
    centers_bbox_ = geometry::bbox3d::from_points(centers_);
  }

  void fit_inequality(const geometry::points3d& points, const common::valuesd& values,
                      const common::valuesd& values_lb, const common::valuesd& values_ub,
                      double absolute_tolerance) {
    if (model_.rbf().nugget() > 0.0)
      throw common::not_supported("RBF with finite nugget");

    auto min_n_points = model_.poly_basis_size() + 1;
    if (points.rows() < min_n_points)
      throw common::invalid_argument("points.rows() >= " + std::to_string(min_n_points));

    if (values.rows() != points.rows())
      throw common::invalid_argument("values.rows() == points.rows()");

    if (values_lb.rows() != points.rows())
      throw common::invalid_argument("values_lb.rows() == points.rows()");

    if (values_ub.rows() != points.rows())
      throw common::invalid_argument("values_ub.rows() == points.rows()");

    if (absolute_tolerance <= 0.0)
      throw common::invalid_argument("absolute_tolerance > 0.0");

    clear_centers();

    auto transformed = affine_transform_points(points);
    interpolation::rbf_inequality_fitter fitter(model_, transformed);

    std::vector<size_t> center_indices;
    std::tie(center_indices, weights_) = fitter.fit(values, values_lb, values_ub, absolute_tolerance);

    centers_ = common::take_rows(transformed, center_indices);
    centers_bbox_ = geometry::bbox3d::from_points(centers_);
  }

  geometry::affine_transform3d point_transform() const {
    return point_transform_;
  }

  void set_evaluation_bbox_impl(const geometry::bbox3d& bbox) {
    auto transformed_bbox = bbox
      .transform(point_transform_)
      .union_hull(centers_bbox_);

    evaluator_ = std::make_unique<interpolation::rbf_evaluator<>>(model_, centers_, transformed_bbox);
    evaluator_->set_weights(weights_);
  }

  void set_point_transform(const geometry::affine_transform3d& affine) {
    point_transform_ = affine;
  }

  const common::valuesd& weights() const {
    return weights_;
  }

private:
  geometry::points3d affine_transform_points(const geometry::points3d& points) const {
    if (point_transform_.is_identity())
      return points;

    geometry::points3d transformed(points.rows(), 3);

    auto transformed_it = common::row_begin(transformed);
    for (auto p : common::row_range(points)) {
      *transformed_it++ = point_transform_.transform_point(p);
    }

    return transformed;
  }

  void clear_centers() {
    centers_ = geometry::points3d();
    centers_bbox_ = geometry::bbox3d();
    weights_ = common::valuesd();
  }

  const model model_;

  geometry::affine_transform3d point_transform_;

  geometry::points3d centers_;
  geometry::bbox3d centers_bbox_;
  common::valuesd weights_;

  std::unique_ptr<interpolation::rbf_evaluator<>> evaluator_;
};

}  // namespace polatory
