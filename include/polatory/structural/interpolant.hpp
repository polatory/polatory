#pragma once

#include <algorithm>
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
  using Points = geometry::Points3;
  using StandardInterpolant = Interpolant<3>;

 public:
  explicit StructuralInterpolant3(const Model& base_model, double outside_value = -1.0,
                                  double blend_power = 1.0)
      : base_model_(base_model), outside_value_(outside_value), blend_power_(blend_power) {
    if (!(blend_power_ > 0.0)) {
      throw std::invalid_argument("blend_power must be positive");
    }
  }

  const Bbox& bbox() const {
    throw_if_not_fitted();
    return bbox_;
  }

  double blend_power() const { return blend_power_; }

  Index num_domains() const { return static_cast<Index>(domains_.size()); }

  double outside_value() const { return outside_value_; }

  void fit(const Points& points, const VecX& values, const std::vector<DomainSpec3>& domain_specs,
           double tolerance, int max_iter = 100, double accuracy = kInfinity) {
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

      for (Index local_i = 0; local_i < static_cast<Index>(indices.size()); ++local_i) {
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

      auto local_interpolant = std::make_unique<StandardInterpolant>(local_model);
      local_interpolant->fit(local_points, local_values, tolerance, max_iter, accuracy);

      domains_.push_back(Domain{spec, std::move(local_interpolant)});
      bbox_ = bbox_.is_empty() ? spec.bbox() : bbox_.convex_hull(spec.bbox());
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
      for (Index i = 0; i < static_cast<Index>(active_indices.size()); ++i) {
        active_points.row(i) = points.row(active_indices.at(static_cast<std::size_t>(i)));
      }

      VecX predictions = domain.interpolant->evaluate_impl(active_points);
      for (Index i = 0; i < static_cast<Index>(active_indices.size()); ++i) {
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

  void set_evaluation_bbox_impl(const Bbox& bbox, double accuracy = kInfinity) {
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
  };

  double box_weight(const geometry::Point3& point, const Bbox& bbox) const {
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
          std::min(point(axis) - bbox.min()(axis), bbox.max()(axis) - point(axis));
      auto u = std::clamp(2.0 * distance_to_face / width(axis), 0.0, 1.0);
      auto smooth = u * u * (3.0 - 2.0 * u);
      weight *= smooth;
    }

    return std::pow(weight, blend_power_);
  }

  void clear() {
    fitted_ = false;
    domains_.clear();
    bbox_ = Bbox();
  }

  void throw_if_not_fitted() const {
    if (!fitted_) {
      throw std::runtime_error("structural interpolant has not been fitted");
    }
  }

  Model base_model_;
  double outside_value_;
  double blend_power_;
  bool fitted_{};
  std::vector<Domain> domains_;
  Bbox bbox_;
};

}  // namespace polatory::structural
