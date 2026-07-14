#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/structural/domain_spec.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::structural {

using TriangleFaces3 =
    Eigen::Matrix<Index, Eigen::Dynamic, 3, Eigen::RowMajor>;

enum class StructuralTrendType {
  kStrongestAlongInputs,
  kBlending,
  kNonDecaying,
};

class StructuralTrendInput3 {
 public:
  StructuralTrendInput3(geometry::Points3 vertices, TriangleFaces3 faces,
                        double strength, double range)
      : vertices_(std::move(vertices)),
        faces_(std::move(faces)),
        strength_(strength),
        range_(range) {
    if (vertices_.rows() == 0) {
      throw std::invalid_argument("trend input vertices must not be empty");
    }
    if (faces_.rows() == 0) {
      throw std::invalid_argument("trend input faces must not be empty");
    }
    if (!(strength_ >= 1.0)) {
      throw std::invalid_argument("trend input strength must be at least 1");
    }
    if (!(range_ > 0.0)) {
      throw std::invalid_argument("trend input range must be positive");
    }

    for (Index i = 0; i < faces_.rows(); ++i) {
      for (Index j = 0; j < 3; ++j) {
        auto vertex_i = faces_(i, j);
        if (vertex_i < 0 || vertex_i >= vertices_.rows()) {
          throw std::out_of_range("trend input face index is outside vertices");
        }
      }
    }
  }

  const TriangleFaces3& faces() const { return faces_; }

  double range() const { return range_; }

  double strength() const { return strength_; }

  const geometry::Points3& vertices() const { return vertices_; }

 private:
  geometry::Points3 vertices_;
  TriangleFaces3 faces_;
  double strength_;
  double range_;
};

struct StructuralTrendSamples3 {
  geometry::Points3 normals;
  VecX ratios;
  VecX distances;
  std::vector<Index> dominant_inputs;
  std::vector<Mat3> anisotropies;
};

class StructuralDomainBuilder3 {
  using Bbox = geometry::Bbox3;
  using Point = geometry::Point3;
  using Points = geometry::Points3;

 public:
  explicit StructuralDomainBuilder3(double domain_size = 0.0,
                                    double overlap = 0.0,
                                    Index min_support_points = 4)
      : domain_size_(domain_size),
        overlap_(overlap),
        min_support_points_(min_support_points) {
    if (!(domain_size_ >= 0.0)) {
      throw std::invalid_argument("domain_size must be non-negative");
    }
    if (!(overlap_ >= 0.0)) {
      throw std::invalid_argument("overlap must be non-negative");
    }
    if (min_support_points_ <= 0) {
      throw std::invalid_argument("min_support_points must be positive");
    }
  }

  std::vector<DomainSpec3> build(
      const Points& points, const std::vector<StructuralTrendInput3>& inputs,
      StructuralTrendType trend_type =
          StructuralTrendType::kStrongestAlongInputs,
      const std::vector<double>& model_parameters = {}) const {
    if (points.rows() == 0) {
      throw std::invalid_argument("points must not be empty");
    }
    validate_inputs(inputs);

    auto prepared = prepare_inputs(inputs);

    Point points_min = points.colwise().minCoeff();
    Point points_max = points.colwise().maxCoeff();
    auto widths = points_max - points_min;

    auto max_range = 0.0;
    for (const auto& input : inputs) {
      max_range = std::max(max_range, input.range());
    }

    auto domain_size = domain_size_;
    if (!(domain_size > 0.0)) {
      domain_size = 2.0 * max_range;
      if (!(domain_size > 0.0)) {
        domain_size = std::max({widths(0), widths(1), widths(2)}) / 3.0;
      }
      if (!(domain_size > 0.0)) {
        domain_size = 1.0;
      }
    }

    auto overlap = overlap_;
    if (!(overlap > 0.0)) {
      overlap = trend_type == StructuralTrendType::kNonDecaying
                    ? 0.5 * domain_size
                    : max_range;
    }

    using Cell = std::array<long long, 3>;
    std::map<Cell, std::vector<Index>> cells;

    for (Index i = 0; i < points.rows(); ++i) {
      Cell cell{};
      for (Index axis = 0; axis < 3; ++axis) {
        cell.at(static_cast<std::size_t>(axis)) =
            static_cast<long long>(std::floor(
                (points(i, axis) - points_min(axis)) / domain_size));
      }
      cells[cell].push_back(i);
    }

    std::vector<DomainSpec3> domains;
    domains.reserve(cells.size());

    for (const auto& [cell, core_indices] : cells) {
      Point core_min;
      Point core_max;
      Point sample_point = Point::Zero();

      for (Index axis = 0; axis < 3; ++axis) {
        auto cell_i = static_cast<double>(
            cell.at(static_cast<std::size_t>(axis)));
        core_min(axis) = points_min(axis) + cell_i * domain_size;
        core_max(axis) = std::min(core_min(axis) + domain_size,
                                  points_max(axis) + domain_size * 1e-9);
      }

      for (auto point_i : core_indices) {
        sample_point += points.row(point_i);
      }
      sample_point /= static_cast<double>(core_indices.size());

      auto trend = evaluate_one(sample_point, inputs, prepared, trend_type);

      Point bbox_min = core_min.array() - overlap;
      Point bbox_max = core_max.array() + overlap;
      auto support_indices = points_inside(points, bbox_min, bbox_max);

      if (static_cast<Index>(support_indices.size()) < min_support_points_) {
        expand_to_minimum_support(points, sample_point, domain_size,
                                  support_indices, bbox_min, bbox_max);
      }

      domains.emplace_back(trend.anisotropy, bbox_min, bbox_max,
                           std::move(support_indices), model_parameters);
    }

    return domains;
  }

  StructuralTrendSamples3 sample(
      const Points& query_points,
      const std::vector<StructuralTrendInput3>& inputs,
      StructuralTrendType trend_type =
          StructuralTrendType::kStrongestAlongInputs) const {
    validate_inputs(inputs);

    StructuralTrendSamples3 result;
    result.normals.resize(query_points.rows(), 3);
    result.ratios.resize(query_points.rows());
    result.distances.resize(query_points.rows());
    result.dominant_inputs.resize(
        static_cast<std::size_t>(query_points.rows()));
    result.anisotropies.reserve(
        static_cast<std::size_t>(query_points.rows()));

    auto prepared = prepare_inputs(inputs);

    for (Index i = 0; i < query_points.rows(); ++i) {
      auto trend = evaluate_one(query_points.row(i), inputs, prepared,
                                trend_type);
      result.normals.row(i) = trend.normal;
      result.ratios(i) = trend.ratio;
      result.distances(i) = trend.distance;
      result.dominant_inputs.at(static_cast<std::size_t>(i)) =
          trend.dominant_input;
      result.anisotropies.push_back(trend.anisotropy);
    }

    return result;
  }

  double domain_size() const { return domain_size_; }

  Index min_support_points() const { return min_support_points_; }

  double overlap() const { return overlap_; }

 private:
  struct PreparedInput {
    Points vertex_normals;
    std::unique_ptr<point_cloud::KdTree<3>> tree;
  };

  struct TrendValue {
    Point normal;
    Mat3 anisotropy;
    double ratio;
    double distance;
    Index dominant_input;
  };

  static Mat3 anisotropy_from_normal(const Point& normal, double ratio) {
    Eigen::Vector3d n = normal.transpose();
    n.normalize();

    auto tangent_scale = std::pow(ratio, -1.0 / 3.0);
    auto normal_scale = std::pow(ratio, 2.0 / 3.0);

    Mat3 projector = n * n.transpose();
    return tangent_scale * (Mat3::Identity() - projector) +
           normal_scale * projector;
  }

  static Points compute_vertex_normals(const StructuralTrendInput3& input) {
    Points normals = Points::Zero(input.vertices().rows(), 3);

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

  static std::vector<PreparedInput> prepare_inputs(
      const std::vector<StructuralTrendInput3>& inputs) {
    std::vector<PreparedInput> prepared;
    prepared.reserve(inputs.size());

    for (const auto& input : inputs) {
      PreparedInput item;
      item.vertex_normals = compute_vertex_normals(input);
      item.tree =
          std::make_unique<point_cloud::KdTree<3>>(input.vertices());
      prepared.push_back(std::move(item));
    }

    return prepared;
  }

  static void validate_inputs(
      const std::vector<StructuralTrendInput3>& inputs) {
    if (inputs.empty()) {
      throw std::invalid_argument("structural trend inputs must not be empty");
    }
  }

  TrendValue evaluate_one(
      const Point& point,
      const std::vector<StructuralTrendInput3>& inputs,
      const std::vector<PreparedInput>& prepared,
      StructuralTrendType trend_type) const {
    std::vector<Point> normals(inputs.size());
    std::vector<double> distances(inputs.size());
    std::vector<double> q_values(inputs.size());
    std::vector<double> contributions(inputs.size());

    Index dominant_input = 0;
    auto best_contribution = -1.0;
    auto best_distance = std::numeric_limits<double>::infinity();

    for (Index input_i = 0; input_i < static_cast<Index>(inputs.size());
         ++input_i) {
      std::vector<Index> indices;
      std::vector<double> nearest_distances;
      prepared.at(static_cast<std::size_t>(input_i))
          .tree->knn_search(point, 1, indices, nearest_distances);

      auto vertex_i = indices.at(0);
      auto distance = nearest_distances.at(0);
      auto normal = prepared.at(static_cast<std::size_t>(input_i))
                        .vertex_normals.row(vertex_i);

      auto q = trend_type == StructuralTrendType::kNonDecaying
                   ? 1.0
                   : std::exp(-distance / inputs.at(
                                               static_cast<std::size_t>(input_i))
                                               .range());
      auto contribution =
          (inputs.at(static_cast<std::size_t>(input_i)).strength() - 1.0) * q;

      normals.at(static_cast<std::size_t>(input_i)) = normal;
      distances.at(static_cast<std::size_t>(input_i)) = distance;
      q_values.at(static_cast<std::size_t>(input_i)) = q;
      contributions.at(static_cast<std::size_t>(input_i)) = contribution;

      if (contribution > best_contribution ||
          (contribution == best_contribution && distance < best_distance)) {
        best_contribution = contribution;
        best_distance = distance;
        dominant_input = input_i;
      }
    }

    Point normal = normals.at(static_cast<std::size_t>(dominant_input));
    auto ratio = 1.0 +
                 contributions.at(static_cast<std::size_t>(dominant_input));

    if (trend_type == StructuralTrendType::kBlending && inputs.size() > 1) {
      Point blended = Point::Zero();
      auto weight_sum = 0.0;
      auto q_sum = 0.0;
      auto reference = normal;

      for (Index input_i = 0; input_i < static_cast<Index>(inputs.size());
           ++input_i) {
        auto aligned = normals.at(static_cast<std::size_t>(input_i));
        if (aligned.dot(reference) < 0.0) {
          aligned *= -1.0;
        }

        auto weight = contributions.at(static_cast<std::size_t>(input_i));
        if (!(weight > 0.0)) {
          weight = q_values.at(static_cast<std::size_t>(input_i));
        }

        blended += weight * aligned;
        weight_sum += weight;
        q_sum += q_values.at(static_cast<std::size_t>(input_i));
      }

      if (weight_sum > 0.0 && blended.norm() > 0.0) {
        normal = blended / blended.norm();
      }

      auto contribution_sum = 0.0;
      for (auto contribution : contributions) {
        contribution_sum += contribution;
      }
      ratio = 1.0 + contribution_sum / std::max(1.0, q_sum);
    }

    normal.normalize();

    auto minimum_distance =
        *std::min_element(distances.begin(), distances.end());

    return TrendValue{normal,
                      anisotropy_from_normal(normal, ratio),
                      ratio,
                      minimum_distance,
                      dominant_input};
  }

  static std::vector<Index> points_inside(const Points& points,
                                          const Point& bbox_min,
                                          const Point& bbox_max) {
    std::vector<Index> result;
    result.reserve(static_cast<std::size_t>(points.rows()));

    for (Index i = 0; i < points.rows(); ++i) {
      auto inside = true;
      for (Index axis = 0; axis < 3; ++axis) {
        if (points(i, axis) < bbox_min(axis) ||
            points(i, axis) > bbox_max(axis)) {
          inside = false;
          break;
        }
      }
      if (inside) {
        result.push_back(i);
      }
    }

    return result;
  }

  void expand_to_minimum_support(const Points& points,
                                 const Point& sample_point,
                                 double domain_size,
                                 std::vector<Index>& support_indices,
                                 Point& bbox_min, Point& bbox_max) const {
    auto expansion = 0.5 * domain_size;

    for (int attempt = 0;
         attempt < 8 &&
         static_cast<Index>(support_indices.size()) < min_support_points_;
         ++attempt) {
      bbox_min.array() -= expansion;
      bbox_max.array() += expansion;
      support_indices = points_inside(points, bbox_min, bbox_max);
      expansion *= 1.5;
    }

    if (static_cast<Index>(support_indices.size()) >= min_support_points_) {
      return;
    }

    std::vector<std::pair<double, Index>> nearest;
    nearest.reserve(static_cast<std::size_t>(points.rows()));
    for (Index i = 0; i < points.rows(); ++i) {
      nearest.emplace_back((points.row(i) - sample_point).squaredNorm(), i);
    }
    std::sort(nearest.begin(), nearest.end());

    support_indices.clear();
    auto count = std::min(min_support_points_, points.rows());
    for (Index i = 0; i < count; ++i) {
      auto point_i = nearest.at(static_cast<std::size_t>(i)).second;
      support_indices.push_back(point_i);
      bbox_min = bbox_min.cwiseMin(points.row(point_i));
      bbox_max = bbox_max.cwiseMax(points.row(point_i));
    }
  }

  double domain_size_;
  double overlap_;
  Index min_support_points_;
};

}  // namespace polatory::structural
