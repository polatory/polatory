#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <polatory/structural/adaptive_domain_builder.hpp>
#include <polatory/structural/domain_orientation_average.hpp>
#include <vector>

namespace polatory::structural {

// Experimental single-input refinement for the adaptive partitioner.
//
// The adaptive builder uses all sampled normals to decide where to split, but its
// standard build() assigns each final leaf the trend sampled at only the leaf
// centroid. This helper preserves the adaptive boxes and support memberships while
// replacing that centroid normal with a decay-weighted, sign-invariant axial
// average of the structural normals sampled at the interpolation points in the
// leaf core.
inline std::vector<DomainSpec3>
build_adaptive_orientation_averaged_domains(
    const geometry::Points3& points,
    const std::vector<StructuralTrendInput3>& inputs,
    StructuralTrendType trend_type =
        StructuralTrendType::kStrongestAlongInputs,
    const std::vector<double>& model_parameters = {},
    double overlap = 0.0,
    double orientation_consistency = 0.97,
    double minimum_core_size = 0.0,
    double maximum_core_size = 0.0,
    Index minimum_core_points = 24,
    Index minimum_support_points = 4,
    int maximum_depth = 20) {
  AdaptiveStructuralDomainBuilder3 builder(
      overlap, orientation_consistency, minimum_core_size,
      maximum_core_size, minimum_core_points,
      minimum_support_points, maximum_depth);

  auto center_sampled_domains =
      builder.build(points, inputs, trend_type, model_parameters);

  // Multiple-input orientation combination is still experimental. Keep the
  // validated centre-sampled behaviour there and isolate this refinement to the
  // current single-reference-mesh benchmarks.
  if (inputs.size() != 1) {
    return center_sampled_domains;
  }

  const auto& input = inputs.front();
  auto vertex_normals = detail::averaged_vertex_normals(input);

  auto actual_overlap = overlap;
  if (!(actual_overlap > 0.0)) {
    actual_overlap = input.range();
  }

  std::vector<DomainSpec3> averaged_domains;
  averaged_domains.reserve(center_sampled_domains.size());

  for (const auto& domain : center_sampled_domains) {
    geometry::Point3 core_min =
        domain.bbox().min().array() + actual_overlap;
    geometry::Point3 core_max =
        domain.bbox().max().array() - actual_overlap;

    std::vector<Index> core_indices;
    core_indices.reserve(domain.support_indices().size());
    geometry::Point3 sample_point = geometry::Point3::Zero();

    for (auto point_i : domain.support_indices()) {
      if (detail::point_inside_box(points.row(point_i), core_min, core_max)) {
        core_indices.push_back(point_i);
        sample_point += points.row(point_i);
      }
    }

    if (!core_indices.empty()) {
      sample_point /= static_cast<double>(core_indices.size());
    } else {
      sample_point = 0.5 * (core_min + core_max);
    }

    auto centre_nearest = detail::nearest_trend_vertex(sample_point, input);
    geometry::Point3 centre_normal =
        vertex_normals.row(centre_nearest.index);

    auto centre_q = trend_type == StructuralTrendType::kNonDecaying
                        ? 1.0
                        : std::exp(-centre_nearest.distance / input.range());
    auto ratio = 1.0 + (input.strength() - 1.0) * centre_q;

    Mat3 axial_tensor = Mat3::Zero();
    auto weight_sum = 0.0;

    for (auto point_i : core_indices) {
      auto nearest = detail::nearest_trend_vertex(points.row(point_i), input);
      Eigen::Vector3d normal =
          vertex_normals.row(nearest.index).transpose();
      auto q = trend_type == StructuralTrendType::kNonDecaying
                   ? 1.0
                   : std::exp(-nearest.distance / input.range());
      auto weight = std::max(q, 1e-12);
      axial_tensor += weight * normal * normal.transpose();
      weight_sum += weight;
    }

    geometry::Point3 representative_normal = centre_normal;
    if (weight_sum > 0.0) {
      axial_tensor /= weight_sum;
      Eigen::SelfAdjointEigenSolver<Mat3> solver(axial_tensor);
      if (solver.info() == Eigen::Success) {
        Eigen::Vector3d axis = solver.eigenvectors().col(2);
        Eigen::Vector3d reference = centre_normal.transpose();
        if (axis.dot(reference) < 0.0) {
          axis *= -1.0;
        }
        representative_normal = axis.transpose();
      }
    }

    auto anisotropy =
        detail::anisotropy_from_axis(representative_normal, ratio);

    averaged_domains.emplace_back(
        anisotropy, domain.bbox().min(), domain.bbox().max(),
        domain.support_indices(), domain.model_parameters());
  }

  return averaged_domains;
}

}  // namespace polatory::structural
