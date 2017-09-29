#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "../common/exception.hpp"
#include "../common/vector_view.hpp"
#include "../geometry/affine_transform.hpp"
#include "../geometry/bbox3.hpp"
#include "../interpolation/rbf_evaluator.hpp"
#include "../interpolation/rbf_fitter.hpp"
#include "../interpolation/rbf_incremental_fitter.hpp"
#include "../polynomial/basis_base.hpp"
#include "../rbf/rbf_base.hpp"

namespace polatory {
namespace driver {

class interpolant {
public:
  using points_type = std::vector<Eigen::Vector3d>;
  using values_type = Eigen::VectorXd;

  interpolant(const rbf::rbf_base& rbf, int poly_degree)
    : rbf_(rbf)
    , poly_degree_(poly_degree)
    , point_transform_(Eigen::Matrix4d::Identity()) {
    if (poly_degree < rbf.order_of_definiteness() - 1 || poly_degree > 2)
      throw common::invalid_parameter("rbf.order_of_definiteness() - 1 <= poly_degree <= 2");
  }

  const points_type& centers() const {
    return centers_;
  }

  values_type evaluate_points(const points_type& points) const {
    auto transformed = affine_transform_points(points);

    return evaluator_->evaluate_points(transformed);
  }

  void fit(const points_type& points, const values_type& values, double absolute_tolerance) {
    auto min_n_points = polynomial::basis_base::dimension(poly_degree_) + 1;
    if (points.size() < min_n_points)
      throw common::invalid_parameter("points.size() >= " + std::to_string(min_n_points));

    auto transformed = affine_transform_points(points);

    interpolation::rbf_fitter fitter(rbf_, poly_degree_, transformed);

    centers_ = transformed;
    weights_ = fitter.fit(values, absolute_tolerance);
  }

  void fit_incrementally(const points_type& points, const values_type& values, double absolute_tolerance) {
    auto min_n_points = polynomial::basis_base::dimension(poly_degree_) + 1;
    if (points.size() < min_n_points)
      throw common::invalid_parameter("points.size() >= " + std::to_string(min_n_points));

    auto transformed = affine_transform_points(points);

    interpolation::rbf_incremental_fitter fitter(rbf_, poly_degree_, transformed);

    std::vector<size_t> center_indices;
    std::tie(center_indices, weights_) = fitter.fit(values, absolute_tolerance);

    auto view = common::make_view(transformed, center_indices);
    centers_ = std::vector<Eigen::Vector3d>(view.begin(), view.end());
  }

  void set_evaluation_bbox(const geometry::bbox3d& bbox) {
    auto transformed_bbox = bbox.affine_transform(point_transform_);

    evaluator_ = std::make_unique<interpolation::rbf_evaluator<>>(rbf_, poly_degree_, centers_, transformed_bbox);
    evaluator_->set_weights(weights_);
  }

  void set_point_transform(const Eigen::Matrix4d& affine_transform) {
    point_transform_ = affine_transform;
  }

  const values_type& weights() const {
    return weights_;
  }

private:
  points_type affine_transform_points(const points_type& points) const {
    points_type transformed;
    transformed.reserve(points.size());

    for (const auto& p : points) {
      transformed.push_back(geometry::affine_transform_point(p, point_transform_));
    }

    return transformed;
  }

  const rbf::rbf_base& rbf_;
  const int poly_degree_;

  Eigen::Matrix4d point_transform_;

  std::vector<Eigen::Vector3d> centers_;
  Eigen::VectorXd weights_;

  std::unique_ptr<interpolation::rbf_evaluator<>> evaluator_;
};

} // namespace driver
} // namespace polatory
