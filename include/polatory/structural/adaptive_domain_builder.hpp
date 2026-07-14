#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/kdtree.hpp>
#include <polatory/structural/domain_builder.hpp>
#include <polatory/structural/domain_spec.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::structural {

// Experimental adaptive builder. It keeps the validated mesh sampling,
// anisotropy and overlap rules, but replaces the regular grid with recursive,
// data-balanced spatial partitions. Cells are subdivided when they are too large
// or when their sign-invariant structural normals are not sufficiently coherent.
class AdaptiveStructuralDomainBuilder3 {
  using Point = geometry::Point3;
  using Points = geometry::Points3;

 public:
  explicit AdaptiveStructuralDomainBuilder3(
      double overlap = 0.0,
      double orientation_consistency = 0.97,
      double minimum_core_size = 0.0,
      double maximum_core_size = 0.0,
      Index minimum_core_points = 24,
      Index minimum_support_points = 4,
      int maximum_depth = 20)
      : overlap_(overlap),
        orientation_consistency_(orientation_consistency),
        minimum_core_size_(minimum_core_size),
        maximum_core_size_(maximum_core_size),
        minimum_core_points_(minimum_core_points),
        minimum_support_points_(minimum_support_points),
        maximum_depth_(maximum_depth) {
    if (!(overlap_ >= 0.0)) {
      throw std::invalid_argument("overlap must be non-negative");
    }
    if (!(orientation_consistency_ > 0.0 &&
          orientation_consistency_ <= 1.0)) {
      throw std::invalid_argument(
          "orientation_consistency must be in (0, 1]");
    }
    if (!(minimum_core_size_ >= 0.0)) {
      throw std::invalid_argument("minimum_core_size must be non-negative");
    }
    if (!(maximum_core_size_ >= 0.0)) {
      throw std::invalid_argument("maximum_core_size must be non-negative");
    }
    if (minimum_core_points_ <= 0) {
      throw std::invalid_argument("minimum_core_points must be positive");
    }
    if (minimum_support_points_ <= 0) {
      throw std::invalid_argument("minimum_support_points must be positive");
    }
    if (maximum_depth_ <= 0) {
      throw std::invalid_argument("maximum_depth must be positive");
    }
  }

  std::vector<DomainSpec3> build(
      const Points& points,
      const std::vector<StructuralTrendInput3>& inputs,
      StructuralTrendType trend_type =
          StructuralTrendType::kStrongestAlongInputs,
      const std::vector<double>& model_parameters = {}) const {
    if (points.rows() == 0) {
      throw std::invalid_argument("points must not be empty");
    }
    if (inputs.empty()) {
      throw std::invalid_argument("structural trend inputs must not be empty");
    }

    auto prepared = prepare_inputs(inputs);

    auto maximum_range = 0.0;
    for (const auto& input : inputs) {
      maximum_range = std::max(maximum_range, input.range());
    }

    auto resolved_overlap = overlap_ > 0.0 ? overlap_ : maximum_range;
    auto resolved_minimum_core_size =
        minimum_core_size_ > 0.0 ? minimum_core_size_ : 0.75 * maximum_range;
    auto resolved_maximum_core_size =
        maximum_core_size_ > 0.0 ? maximum_core_size_ : 2.10 * maximum_range;

    if (!(resolved_maximum_core_size > 0.0)) {
      resolved_maximum_core_size = 1.0;
    }
    if (!(resolved_minimum_core_size > 0.0)) {
      resolved_minimum_core_size = 0.25 * resolved_maximum_core_size;
    }
    if (resolved_minimum_core_size > resolved_maximum_core_size) {
      throw std::invalid_argument(
          "minimum_core_size must not exceed maximum_core_size");
    }

    Points sampled_normals(points.rows(), 3);
    for (Index point_i = 0; point_i < points.rows(); ++point_i) {
      sampled_normals.row(point_i) =
          evaluate_one(points.row(point_i), inputs, prepared, trend_type).normal;
    }

    auto active_axes = detect_active_axes(points, sampled_normals);

    Node root;
    root.minimum = points.colwise().minCoeff();
    root.maximum = points.colwise().maxCoeff();
    root.indices.resize(static_cast<std::size_t>(points.rows()));
    std::iota(root.indices.begin(), root.indices.end(), Index{0});

    std::vector<Node> leaves;
    split_recursive(root, points, sampled_normals, active_axes,
                    resolved_minimum_core_size,
                    resolved_maximum_core_size, 0, leaves);

    std::vector<DomainSpec3> domains;
    domains.reserve(leaves.size());

    for (const auto& leaf : leaves) {
      Point sample_point = Point::Zero();
      for (auto point_i : leaf.indices) {
        sample_point += points.row(point_i);
      }
      sample_point /= static_cast<double>(leaf.indices.size());

      auto trend = evaluate_one(sample_point, inputs, prepared, trend_type);

      Point bbox_minimum = leaf.minimum.array() - resolved_overlap;
      Point bbox_maximum = leaf.maximum.array() + resolved_overlap;
      auto support_indices =
          points_inside(points, bbox_minimum, bbox_maximum);

      if (static_cast<Index>(support_indices.size()) <
          minimum_support_points_) {
        expand_to_minimum_support(points, sample_point,
                                  resolved_maximum_core_size,
                                  support_indices, bbox_minimum,
                                  bbox_maximum);
      }

      domains.emplace_back(trend.anisotropy, bbox_minimum, bbox_maximum,
                           std::move(support_indices), model_parameters);
    }

    return domains;
  }

  double overlap() const { return overlap_; }
  double orientation_consistency() const {
    return orientation_consistency_;
  }
  double minimum_core_size() const { return minimum_core_size_; }
  double maximum_core_size() const { return maximum_core_size_; }
  Index minimum_core_points() const { return minimum_core_points_; }
  Index minimum_support_points() const { return minimum_support_points_; }
  int maximum_depth() const { return maximum_depth_; }

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

  struct Node {
    Point minimum;
    Point maximum;
    std::vector<Index> indices;
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
                   : std::exp(-distance /
                              inputs.at(static_cast<std::size_t>(input_i))
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
    auto ratio =
        1.0 + contributions.at(static_cast<std::size_t>(dominant_input));

    if (trend_type == StructuralTrendType::kBlending && inputs.size() > 1) {
      Point blended = Point::Zero();
      auto weight_sum = 0.0;
      auto q_sum = 0.0;
      auto reference = normal;

      for (Index input_i = 0;
           input_i < static_cast<Index>(inputs.size()); ++input_i) {
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

  static double axial_consistency(const std::vector<Index>& indices,
                                  const Points& normals) {
    if (indices.empty()) {
      return 1.0;
    }

    Mat3 tensor = Mat3::Zero();
    for (auto point_i : indices) {
      Eigen::Vector3d normal = normals.row(point_i).transpose();
      tensor += normal * normal.transpose();
    }
    tensor /= static_cast<double>(indices.size());

    Eigen::SelfAdjointEigenSolver<Mat3> solver(tensor);
    if (solver.info() != Eigen::Success) {
      return 1.0;
    }

    auto trace = solver.eigenvalues().sum();
    if (!(trace > 0.0)) {
      return 1.0;
    }

    return solver.eigenvalues()(2) / trace;
  }

  static std::array<bool, 3> detect_active_axes(
      const Points& points, const Points& normals) {
    constexpr Index kNumBins = 8;
    constexpr double kVariationThreshold = 0.02;

    std::array<bool, 3> active_axes{false, false, false};

    Mat3 global_tensor = Mat3::Zero();
    for (Index point_i = 0; point_i < normals.rows(); ++point_i) {
      Eigen::Vector3d normal = normals.row(point_i).transpose();
      global_tensor += normal * normal.transpose();
    }
    global_tensor /= static_cast<double>(normals.rows());

    for (Index axis = 0; axis < 3; ++axis) {
      auto coordinate_minimum = points.col(axis).minCoeff();
      auto coordinate_maximum = points.col(axis).maxCoeff();
      auto span = coordinate_maximum - coordinate_minimum;
      if (!(span > 0.0)) {
        continue;
      }

      std::array<Mat3, kNumBins> bin_tensors;
      std::array<Index, kNumBins> bin_counts{};
      for (auto& tensor : bin_tensors) {
        tensor.setZero();
      }

      for (Index point_i = 0; point_i < points.rows(); ++point_i) {
        auto normalized =
            (points(point_i, axis) - coordinate_minimum) / span;
        auto bin_i = static_cast<Index>(
            std::floor(normalized * static_cast<double>(kNumBins)));
        bin_i = std::clamp<Index>(bin_i, 0, kNumBins - 1);

        Eigen::Vector3d normal = normals.row(point_i).transpose();
        bin_tensors.at(static_cast<std::size_t>(bin_i)) +=
            normal * normal.transpose();
        ++bin_counts.at(static_cast<std::size_t>(bin_i));
      }

      auto maximum_variation = 0.0;
      for (Index bin_i = 0; bin_i < kNumBins; ++bin_i) {
        auto count = bin_counts.at(static_cast<std::size_t>(bin_i));
        if (count == 0) {
          continue;
        }
        auto local_tensor =
            bin_tensors.at(static_cast<std::size_t>(bin_i)) /
            static_cast<double>(count);
        maximum_variation =
            std::max(maximum_variation,
                     (local_tensor - global_tensor).norm());
      }

      active_axes.at(static_cast<std::size_t>(axis)) =
          maximum_variation > kVariationThreshold;
    }

    return active_axes;
  }

  static bool split_indices(const Node& node, Index axis,
                            const Points& points, Node& left,
                            Node& right) {
    if (node.indices.size() < 2) {
      return false;
    }

    std::vector<double> coordinates;
    coordinates.reserve(node.indices.size());
    for (auto point_i : node.indices) {
      coordinates.push_back(points(point_i, axis));
    }

    auto middle = coordinates.begin() +
                  static_cast<std::ptrdiff_t>(coordinates.size() / 2);
    std::nth_element(coordinates.begin(), middle, coordinates.end());
    auto split_value = *middle;

    left.minimum = node.minimum;
    left.maximum = node.maximum;
    right.minimum = node.minimum;
    right.maximum = node.maximum;
    left.maximum(axis) = split_value;
    right.minimum(axis) = split_value;

    left.indices.reserve(node.indices.size() / 2 + 1);
    right.indices.reserve(node.indices.size() / 2 + 1);

    for (auto point_i : node.indices) {
      if (points(point_i, axis) < split_value) {
        left.indices.push_back(point_i);
      } else {
        right.indices.push_back(point_i);
      }
    }

    if (left.indices.empty() || right.indices.empty()) {
      return false;
    }

    return true;
  }

  Index choose_split_axis(
      const Node& node, const Points& points, const Points& normals,
      const std::array<bool, 3>& active_axes, bool oversized) const {
    auto parent_consistency = axial_consistency(node.indices, normals);
    auto best_score = -std::numeric_limits<double>::infinity();
    Index best_axis = -1;

    for (Index axis = 0; axis < 3; ++axis) {
      if (!active_axes.at(static_cast<std::size_t>(axis))) {
        continue;
      }

      auto span = node.maximum(axis) - node.minimum(axis);
      if (!(span > 0.0)) {
        continue;
      }

      Node left;
      Node right;
      if (!split_indices(node, axis, points, left, right)) {
        continue;
      }
      if (static_cast<Index>(left.indices.size()) < minimum_core_points_ ||
          static_cast<Index>(right.indices.size()) < minimum_core_points_) {
        continue;
      }

      auto left_consistency = axial_consistency(left.indices, normals);
      auto right_consistency = axial_consistency(right.indices, normals);
      auto weighted_child_consistency =
          (left_consistency * static_cast<double>(left.indices.size()) +
           right_consistency * static_cast<double>(right.indices.size())) /
          static_cast<double>(node.indices.size());

      auto improvement = weighted_child_consistency - parent_consistency;
      auto score = oversized ? span + improvement * span : improvement * span;

      if (score > best_score) {
        best_score = score;
        best_axis = axis;
      }
    }

    return best_axis;
  }

  void split_recursive(
      const Node& node, const Points& points, const Points& normals,
      const std::array<bool, 3>& active_axes,
      double minimum_core_size, double maximum_core_size,
      int depth, std::vector<Node>& leaves) const {
    auto maximum_active_span = 0.0;
    for (Index axis = 0; axis < 3; ++axis) {
      if (active_axes.at(static_cast<std::size_t>(axis))) {
        maximum_active_span =
            std::max(maximum_active_span,
                     node.maximum(axis) - node.minimum(axis));
      }
    }

    auto consistency = axial_consistency(node.indices, normals);
    auto oversized = maximum_active_span > maximum_core_size;
    auto inconsistent =
        consistency < orientation_consistency_ &&
        maximum_active_span > minimum_core_size;
    auto enough_points =
        static_cast<Index>(node.indices.size()) >=
        2 * minimum_core_points_;

    if (depth >= maximum_depth_ || !enough_points ||
        (!oversized && !inconsistent)) {
      leaves.push_back(node);
      return;
    }

    auto split_axis =
        choose_split_axis(node, points, normals, active_axes, oversized);
    if (split_axis < 0) {
      leaves.push_back(node);
      return;
    }

    Node left;
    Node right;
    if (!split_indices(node, split_axis, points, left, right)) {
      leaves.push_back(node);
      return;
    }

    split_recursive(left, points, normals, active_axes,
                    minimum_core_size, maximum_core_size,
                    depth + 1, leaves);
    split_recursive(right, points, normals, active_axes,
                    minimum_core_size, maximum_core_size,
                    depth + 1, leaves);
  }

  static std::vector<Index> points_inside(const Points& points,
                                          const Point& bbox_minimum,
                                          const Point& bbox_maximum) {
    std::vector<Index> result;
    result.reserve(static_cast<std::size_t>(points.rows()));

    for (Index point_i = 0; point_i < points.rows(); ++point_i) {
      auto inside = true;
      for (Index axis = 0; axis < 3; ++axis) {
        if (points(point_i, axis) < bbox_minimum(axis) ||
            points(point_i, axis) > bbox_maximum(axis)) {
          inside = false;
          break;
        }
      }
      if (inside) {
        result.push_back(point_i);
      }
    }

    return result;
  }

  void expand_to_minimum_support(
      const Points& points, const Point& sample_point,
      double reference_size, std::vector<Index>& support_indices,
      Point& bbox_minimum, Point& bbox_maximum) const {
    auto expansion = 0.5 * reference_size;

    for (int attempt = 0;
         attempt < 8 &&
         static_cast<Index>(support_indices.size()) <
             minimum_support_points_;
         ++attempt) {
      bbox_minimum.array() -= expansion;
      bbox_maximum.array() += expansion;
      support_indices =
          points_inside(points, bbox_minimum, bbox_maximum);
      expansion *= 1.5;
    }

    if (static_cast<Index>(support_indices.size()) >=
        minimum_support_points_) {
      return;
    }

    std::vector<std::pair<double, Index>> nearest;
    nearest.reserve(static_cast<std::size_t>(points.rows()));
    for (Index point_i = 0; point_i < points.rows(); ++point_i) {
      nearest.emplace_back(
          (points.row(point_i) - sample_point).squaredNorm(), point_i);
    }
    std::sort(nearest.begin(), nearest.end());

    support_indices.clear();
    auto count = std::min(minimum_support_points_, points.rows());
    for (Index i = 0; i < count; ++i) {
      auto point_i = nearest.at(static_cast<std::size_t>(i)).second;
      support_indices.push_back(point_i);
      bbox_minimum = bbox_minimum.cwiseMin(points.row(point_i));
      bbox_maximum = bbox_maximum.cwiseMax(points.row(point_i));
    }
  }

  double overlap_;
  double orientation_consistency_;
  double minimum_core_size_;
  double maximum_core_size_;
  Index minimum_core_points_;
  Index minimum_support_points_;
  int maximum_depth_;
};

}  // namespace polatory::structural
