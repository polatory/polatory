#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include <polatory/common/exception.hpp>
#include <polatory/common/vector_view.hpp>
#include <polatory/geometry/affine_transform3d.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/interpolation/rbf_evaluator.hpp>
#include <polatory/interpolation/rbf_fitter.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/polynomial/basis_base.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory {

class interpolant {
public:
  using points_type = std::vector<Eigen::Vector3d>;
  using values_type = Eigen::VectorXd;

  interpolant(const rbf::rbf_base& rbf, int poly_dimension, int poly_degree)
    : rbf_(rbf)
    , poly_dimension_(poly_dimension)
    , poly_degree_(poly_degree) {
    if (poly_degree < rbf.order_of_cpd() - 1 || poly_degree > 2)
      throw common::invalid_argument("rbf.order_of_cpd() - 1 <= poly_degree <= 2");
  }

  const points_type& centers() const {
    return centers_;
  }

  geometry::bbox3d centers_bbox() const {
    return centers_bbox_;
  }

  values_type evaluate_points(const points_type& points) {
    set_evaluation_bbox_impl(geometry::bbox3d::from_points(points));

    return evaluate_points_impl(points);
  }

  values_type evaluate_points_impl(const points_type& points) const {
    auto transformed = affine_transform_points(points);

    return evaluator_->evaluate_points(transformed);
  }

  void fit(const points_type& points, const values_type& values, double absolute_tolerance) {
    auto min_n_points = polynomial::basis_base::basis_size(poly_dimension_, poly_degree_) + 1;
    if (points.size() < min_n_points)
      throw common::invalid_argument("points.size() >= " + std::to_string(min_n_points));

    auto transformed = affine_transform_points(points);

    interpolation::rbf_fitter fitter(rbf_, poly_dimension_, poly_degree_, transformed);

    centers_ = transformed;
    centers_bbox_ = geometry::bbox3d::from_points(centers_);
    weights_ = fitter.fit(values, absolute_tolerance);
  }

  void fit_incrementally(const points_type& points, const values_type& values, double absolute_tolerance) {
    auto min_n_points = polynomial::basis_base::basis_size(poly_dimension_, poly_degree_) + 1;
    if (points.size() < min_n_points)
      throw common::invalid_argument("points.size() >= " + std::to_string(min_n_points));

    auto transformed = affine_transform_points(points);

    interpolation::rbf_incremental_fitter fitter(rbf_, poly_dimension_, poly_degree_, transformed);

    std::vector<size_t> center_indices;
    std::tie(center_indices, weights_) = fitter.fit(values, absolute_tolerance);

    auto view = common::make_view(transformed, center_indices);
    centers_ = std::vector<Eigen::Vector3d>(view.begin(), view.end());
    centers_bbox_ = geometry::bbox3d::from_points(centers_);
  }

  geometry::affine_transform3d point_transform() const {
    return point_transform_;
  }

  void set_evaluation_bbox_impl(const geometry::bbox3d& bbox) {
    auto transformed_bbox = bbox
      .transform(point_transform_)
      .union_hull(centers_bbox_);

    evaluator_ = std::make_unique<interpolation::rbf_evaluator<>>(rbf_, poly_dimension_, poly_degree_, centers_, transformed_bbox);
    evaluator_->set_weights(weights_);
  }

  void set_point_transform(const geometry::affine_transform3d& affine) {
    point_transform_ = affine;
  }

  const values_type& weights() const {
    return weights_;
  }

private:
  points_type affine_transform_points(const points_type& points) const {
    if (point_transform_.is_identity())
      return points;

    points_type transformed;
    transformed.reserve(points.size());

    for (const auto& p : points) {
      transformed.push_back(point_transform_.transform_point(p));
    }

    return transformed;
  }

  const rbf::rbf_base& rbf_;
  const int poly_dimension_;
  const int poly_degree_;

  geometry::affine_transform3d point_transform_;

  std::vector<Eigen::Vector3d> centers_;
  geometry::bbox3d centers_bbox_;
  Eigen::VectorXd weights_;

  std::unique_ptr<interpolation::rbf_evaluator<>> evaluator_;
};

} // namespace polatory
