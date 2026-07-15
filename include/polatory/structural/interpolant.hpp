#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolant.hpp>
#include <polatory/model.hpp>
#include <polatory/structural/domain_spec.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::structural {

class StructuralInterpolant3 {
  static constexpr double kInfinity = std::numeric_limits<double>::infinity();
  using Bbox = geometry::Bbox3;
  using Model = Model<3>;
  using Point = geometry::Point3;
  using Points = geometry::Points3;
  using StandardInterpolant = Interpolant<3>;

 public:
  explicit StructuralInterpolant3(const Model& base_model,
                                  double outside_value = -1.0,
                                  double blend_power = 1.0,
                                  double alignment_strength = 0.0)
      : base_model_(base_model),
        outside_value_(outside_value),
        blend_power_(blend_power),
        alignment_strength_(alignment_strength) {
    if (!(blend_power_ > 0.0)) {
      throw std::invalid_argument("blend_power must be positive");
    }
    if (!(alignment_strength_ >= 0.0)) {
      throw std::invalid_argument("alignment_strength must be non-negative");
    }
  }

  const Bbox& bbox() const {
    throw_if_not_fitted();
    return bbox_;
  }

  double blend_power() const { return blend_power_; }

  double alignment_strength() const { return alignment_strength_; }

  Index num_domains() const { return static_cast<Index>(domains_.size()); }

  double outside_value() const { return outside_value_; }

  std::vector<double> domain_offsets() const {
    throw_if_not_fitted();
    std::vector<double> offsets;
    offsets.reserve(domains_.size());
    for (const auto& domain : domains_) {
      offsets.push_back(domain.offset);
    }
    return offsets;
  }

  void fit(const Points& points, const VecX& values,
           const std::vector<DomainSpec3>& domain_specs,
           double tolerance, int max_iter = 100,
           double accuracy = kInfinity) {
    if (points.rows() != values.rows()) {
      throw std::invalid_argument("values.rows() must equal points.rows()");
    }
    if (points.rows() == 0) {
      throw std::invalid_argument("points must not be empty");
    }
    if (domain_specs.empty()) {
      throw std::invalid_argument("domain_specs must not be empty");
    }
    if (!(tolerance > 0.0)) {
      throw std::invalid_argument("tolerance must be positive");
    }
    if (max_iter < 0) {
      throw std::invalid_argument("max_iter must be nonnegative");
    }
    if (!(accuracy > 0.0)) {
      throw std::invalid_argument("accuracy must be positive");
    }

    clear();

    for (const auto& spec : domain_specs) {
      const auto& indices = spec.support_indices();
      Points local_points(static_cast<Index>(indices.size()), 3);
      VecX local_values(static_cast<Index>(indices.size()));

      for (Index local_i = 0;
           local_i < static_cast<Index>(indices.size()); ++local_i) {
        auto global_i = indices.at(static_cast<std::size_t>(local_i));
        if (global_i < 0 || global_i >= points.rows()) {
          throw std::out_of_range("domain support index is outside points");
        }
        local_points.row(local_i) = points.row(global_i);
        local_values(local_i) = values(global_i);
      }

      Model local_model = base_model_;
      if (!spec.model_parameters().empty()) {
        local_model.set_parameters(spec.model_parameters());
      }
      for (auto& rbf : local_model.rbfs()) {
        rbf.set_anisotropy(spec.anisotropy());
      }

      auto local_interpolant =
          std::make_unique<StandardInterpolant>(local_model);
      local_interpolant->fit(local_points, local_values, tolerance,
                              max_iter, accuracy);

      domains_.push_back(
          Domain{spec, std::move(local_interpolant), 0.0});
      bbox_ = bbox_.is_empty() ? spec.bbox()
                               : bbox_.convex_hull(spec.bbox());
    }

    if (alignment_strength_ > 0.0 && domains_.size() > 1) {
      align_domain_offsets(accuracy);
    }

    fitted_ = true;
  }

  VecX evaluate(const Points& points, double accuracy = kInfinity) {
    throw_if_not_fitted();
    if (!(accuracy > 0.0)) {
      throw std::invalid_argument("accuracy must be positive");
    }
    if (points.rows() == 0) {
      return VecX();
    }

    set_evaluation_bbox_impl(Bbox::from_points(points), accuracy);
    return evaluate_impl(points);
  }

  VecX evaluate_impl(const Points& points) const {
    throw_if_not_fitted();

    VecX numerator = VecX::Zero(points.rows());
    VecX denominator = VecX::Zero(points.rows());

    for (const auto& domain : domains_) {
      std::vector<Index> active_indices;
      std::vector<double> active_weights;
      active_indices.reserve(static_cast<std::size_t>(points.rows()));
      active_weights.reserve(static_cast<std::size_t>(points.rows()));

      for (Index i = 0; i < points.rows(); ++i) {
        auto weight = box_weight(points.row(i), domain.spec.bbox());
        if (weight > 0.0) {
          active_indices.push_back(i);
          active_weights.push_back(weight);
        }
      }

      if (active_indices.empty()) {
        continue;
      }

      Points active_points(static_cast<Index>(active_indices.size()), 3);
      for (Index i = 0;
           i < static_cast<Index>(active_indices.size()); ++i) {
        active_points.row(i) =
            points.row(active_indices.at(static_cast<std::size_t>(i)));
      }

      VecX predictions = domain.interpolant->evaluate_impl(active_points);
      predictions.array() += domain.offset;
      for (Index i = 0;
           i < static_cast<Index>(active_indices.size()); ++i) {
        auto query_i = active_indices.at(static_cast<std::size_t>(i));
        auto weight = active_weights.at(static_cast<std::size_t>(i));
        numerator(query_i) += weight * predictions(i);
        denominator(query_i) += weight;
      }
    }

    VecX result = VecX::Constant(points.rows(), outside_value_);
    for (Index i = 0; i < points.rows(); ++i) {
      if (denominator(i) > 0.0) {
        result(i) = numerator(i) / denominator(i);
      }
    }
    return result;
  }

  void set_evaluation_bbox_impl(const Bbox& bbox,
                                double accuracy = kInfinity) {
    throw_if_not_fitted();
    if (!(accuracy > 0.0)) {
      throw std::invalid_argument("accuracy must be positive");
    }

    for (auto& domain : domains_) {
      domain.interpolant->set_evaluation_bbox_impl(bbox, accuracy);
    }
  }

 private:
  struct Domain {
    DomainSpec3 spec;
    std::unique_ptr<StandardInterpolant> interpolant;
    double offset;
  };

  double box_weight(const Point& point, const Bbox& bbox) const {
    if (!bbox.contains(point)) {
      return 0.0;
    }

    double weight = 1.0;
    auto width = bbox.width();
    for (Index axis = 0; axis < 3; ++axis) {
      if (!(width(axis) > 0.0)) {
        return 0.0;
      }

      auto distance_to_face =
          std::min(point(axis) - bbox.min()(axis),
                   bbox.max()(axis) - point(axis));
      auto u = std::clamp(2.0 * distance_to_face / width(axis),
                          0.0, 1.0);
      auto smooth = u * u * (3.0 - 2.0 * u);
      weight *= smooth;
    }

    return std::pow(weight, blend_power_);
  }

  static bool overlap_bbox(const Bbox& a, const Bbox& b,
                           Point& overlap_min, Point& overlap_max) {
    overlap_min = a.min().cwiseMax(b.min());
    overlap_max = a.max().cwiseMin(b.max());
    return (overlap_max.array() > overlap_min.array()).all();
  }

  static Points make_overlap_samples(const Point& overlap_min,
                                     const Point& overlap_max) {
    constexpr std::array<double, 3> fractions{0.2, 0.5, 0.8};
    Points samples(27, 3);
    Index sample_i = 0;
    for (auto fx : fractions) {
      for (auto fy : fractions) {
        for (auto fz : fractions) {
          samples.row(sample_i) =
              overlap_min.array() +
              Point(fx, fy, fz).array() *
                  (overlap_max - overlap_min).array();
          ++sample_i;
        }
      }
    }
    return samples;
  }

  void align_domain_offsets(double accuracy) {
    auto n = static_cast<Index>(domains_.size());
    Eigen::MatrixXd system = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
    Index edge_count = 0;

    for (Index i = 0; i < n; ++i) {
      for (Index j = i + 1; j < n; ++j) {
        Point overlap_min;
        Point overlap_max;
        if (!overlap_bbox(domains_.at(static_cast<std::size_t>(i)).spec.bbox(),
                          domains_.at(static_cast<std::size_t>(j)).spec.bbox(),
                          overlap_min, overlap_max)) {
          continue;
        }

        auto samples = make_overlap_samples(overlap_min, overlap_max);
        auto predictions_i =
            domains_.at(static_cast<std::size_t>(i))
                .interpolant->evaluate(samples, accuracy);
        auto predictions_j =
            domains_.at(static_cast<std::size_t>(j))
                .interpolant->evaluate(samples, accuracy);

        auto weighted_difference = 0.0;
        auto weight_sum = 0.0;
        for (Index sample_i = 0; sample_i < samples.rows(); ++sample_i) {
          auto wi = box_weight(
              samples.row(sample_i),
              domains_.at(static_cast<std::size_t>(i)).spec.bbox());
          auto wj = box_weight(
              samples.row(sample_i),
              domains_.at(static_cast<std::size_t>(j)).spec.bbox());
          auto overlap_weight = std::sqrt(wi * wj);

          // Give the zero-level neighbourhood more influence while retaining
          // information from the complete overlap volume.
          auto level_weight =
              1.0 / (1.0 + std::abs(predictions_i(sample_i)) +
                     std::abs(predictions_j(sample_i)));
          auto weight = overlap_weight * level_weight;
          weighted_difference +=
              weight * (predictions_j(sample_i) -
                        predictions_i(sample_i));
          weight_sum += weight;
        }

        if (!(weight_sum > 1e-12)) {
          continue;
        }

        auto difference = weighted_difference / weight_sum;
        auto confidence = weight_sum /
                          static_cast<double>(samples.rows());

        system(i, i) += confidence;
        system(j, j) += confidence;
        system(i, j) -= confidence;
        system(j, i) -= confidence;
        rhs(i) += confidence * difference;
        rhs(j) -= confidence * difference;
        ++edge_count;
      }
    }

    if (edge_count == 0) {
      return;
    }

    // The pairwise system is invariant to a common global offset. A tiny ridge
    // gives disconnected overlap components a stable minimum-norm solution.
    system.diagonal().array() += 1e-10;
    Eigen::VectorXd offsets = system.ldlt().solve(rhs);
    if (!offsets.allFinite()) {
      return;
    }

    // Preserve the model's global zero reference rather than translating every
    // local field together.
    offsets.array() -= offsets.mean();

    for (Index i = 0; i < n; ++i) {
      domains_.at(static_cast<std::size_t>(i)).offset =
          alignment_strength_ * offsets(i);
    }
  }

  void clear() {
    fitted_ = false;
    domains_.clear();
    bbox_ = Bbox();
  }

  void throw_if_not_fitted() const {
    if (!fitted_) {
      throw std::runtime_error(
          "structural interpolant has not been fitted");
    }
  }

  Model base_model_;
  double outside_value_;
  double blend_power_;
  double alignment_strength_;
  bool fitted_{};
  std::vector<Domain> domains_;
  Bbox bbox_;
};

}  // namespace polatory::structural
