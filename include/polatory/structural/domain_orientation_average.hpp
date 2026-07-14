#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <polatory/structural/domain_builder.hpp>
#include <utility>
#include <vector>

namespace polatory::structural {

namespace detail {

struct NearestTrendVertex3 {
  Index index;
  double distance;
};

inline NearestTrendVertex3 nearest_trend_vertex(
    const geometry::Point3& point, const StructuralTrendInput3& input) {
  auto best_index = Index{0};
  auto best_squared_distance = std::numeric_limits<double>::infinity();

  for (Index vertex_i = 0; vertex_i < input.vertices().rows(); ++vertex_i) {
    auto squared_distance =
        (input.vertices().row(vertex_i) - point).squaredNorm();
    if (squared_distance < best_squared_distance) {
      best_squared_distance = squared_distance;
      best_index = vertex_i;
    }
  }

  return NearestTrendVertex3{best_index, std::sqrt(best_squared_distance)};
}

inline geometry::Points3 averaged_vertex_normals(
    const StructuralTrendInput3& input) {
  geometry::Points3 normals =
      geometry::Points3::Zero(input.vertices().rows(), 3);

  for (Index face_i = 0; face_i < input.faces().rows(); ++face_i) {
    auto a = input.faces()(face_i, 0);
    auto b = input.faces()(face_i, 1);
    auto c = input.faces()(face_i, 2);

    Eigen::Vector3d va = input.vertices().row(a).transpose();
    Eigen::Vector3d vb = input.vertices().row(b).transpose();
    Eigen::Vector3d vc = input.vertices().row(c).transpose();

    Eigen::Vector3d face_normal = (vb - va).cross(vc - va);
    auto norm = face_normal.norm();
    if (!(norm > 0.0)) {
      continue;
    }
    face_normal /= norm;

    normals.row(a) += face_normal.transpose();
    normals.row(b) += face_normal.transpose();
    normals.row(c) += face_normal.transpose();
  }

  for (Index vertex_i = 0; vertex_i < normals.rows(); ++vertex_i) {
    auto norm = normals.row(vertex_i).norm();
    if (norm > 0.0) {
      normals.row(vertex_i) /= norm;
    } else {
      normals.row(vertex_i) << 0.0, 0.0, 1.0;
    }
  }

  return normals;
}

inline bool point_inside_box(const geometry::Point3& point,
                             const geometry::Point3& box_min,
                             const geometry::Point3& box_max) {
  for (Index axis = 0; axis < 3; ++axis) {
    if (point(axis) < box_min(axis) || point(axis) > box_max(axis)) {
      return false;
    }
  }
  return true;
}

inline Mat3 anisotropy_from_axis(const geometry::Point3& normal,
                                 double ratio) {
  Eigen::Vector3d n = normal.transpose();
  n.normalize();

  auto tangent_scale = std::pow(ratio, -1.0 / 3.0);
  auto normal_scale = std::pow(ratio, 2.0 / 3.0);
  Mat3 projector = n * n.transpose();

  return tangent_scale * (Mat3::Identity() - projector) +
         normal_scale * projector;
}

inline double resolved_domain_size(
    double requested_domain_size,
    const std::vector<StructuralTrendInput3>& inputs,
    const geometry::Points3& points) {
  if (requested_domain_size > 0.0) {
    return requested_domain_size;
  }

  auto max_range = 0.0;
  for (const auto& input : inputs) {
    max_range = std::max(max_range, input.range());
  }

  auto size = 1.25 * max_range;
  if (!(size > 0.0)) {
    auto point_min = points.colwise().minCoeff();
    auto point_max = points.colwise().maxCoeff();
    auto widths = point_max - point_min;
    size = std::max({widths(0), widths(1), widths(2)}) / 4.0;
  }
  return size > 0.0 ? size : 1.0;
}

inline double resolved_overlap(
    double requested_overlap, double domain_size,
    const std::vector<StructuralTrendInput3>& inputs,
    StructuralTrendType trend_type) {
  if (requested_overlap > 0.0) {
    return requested_overlap;
  }

  if (trend_type == StructuralTrendType::kNonDecaying) {
    return 0.5 * domain_size;
  }

  auto max_range = 0.0;
  for (const auto& input : inputs) {
    max_range = std::max(max_range, input.range());
  }
  return max_range;
}

}  // namespace detail

// Experimental single-input refinement used to test whether Leapfrog derives one
// representative domain orientation from several local structural samples rather
// than from only the domain centroid. Domain boxes and support memberships remain
// exactly those produced by StructuralDomainBuilder3; only each domain's normal is
// replaced by a sign-invariant, decay-weighted axial average of the nearest mesh
// vertex normals sampled at the interpolation points in that domain core.
inline std::vector<DomainSpec3> build_orientation_averaged_domains(
    const geometry::Points3& points,
    const std::vector<StructuralTrendInput3>& inputs,
    StructuralTrendType trend_type =
        StructuralTrendType::kStrongestAlongInputs,
    const std::vector<double>& model_parameters = {},
    double domain_size = 0.0, double overlap = 0.0,
    Index min_support_points = 4) {
  StructuralDomainBuilder3 builder(domain_size, overlap, min_support_points);
  auto center_sampled_domains =
      builder.build(points, inputs, trend_type, model_parameters);

  // Multiple-input equations are still being reverse engineered. Preserve the
  // validated centre-sampled behaviour there and isolate this experiment to the
  // current single-reference-mesh benchmark.
  if (inputs.size() != 1) {
    return center_sampled_domains;
  }

  const auto& input = inputs.front();
  auto vertex_normals = detail::averaged_vertex_normals(input);
  auto actual_domain_size =
      detail::resolved_domain_size(domain_size, inputs, points);
  auto actual_overlap = detail::resolved_overlap(
      overlap, actual_domain_size, inputs, trend_type);

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
