#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <polatory/common/complementary_indices.hpp>
#include <polatory/common/zip_sort.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/types.hpp>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::interpolation {

template <class Model>
class rbf_inequality_fitter {
  static constexpr int kDim = Model::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Solver = rbf_solver<Model>;
  using Evaluator = rbf_evaluator<Model>;

 public:
  rbf_inequality_fitter(const Model& model, const Points& points)
      : model_(model),
        points_(points),
        n_points_(points.rows()),
        n_poly_basis_(model.poly_basis_size()),
        bbox_(Bbox::from_points(points)) {}

  std::pair<std::vector<index_t>, common::valuesd> fit(const common::valuesd& values,
                                                       const common::valuesd& values_lb,
                                                       const common::valuesd& values_ub,
                                                       double absolute_tolerance,
                                                       int max_iter) const {
    double filtering_distance = bbox_.width().mean() / 4.0;

    auto not_nan = [](double d) { return !std::isnan(d); };
    auto eq_idcs = arg_where(values, not_nan);
    auto n_eq = static_cast<index_t>(eq_idcs.size());

    auto lb_idcs = arg_where(values_lb, not_nan);
    auto ub_idcs = arg_where(values_ub, not_nan);
    std::vector<index_t> ineq_idcs;
    std::set_union(lb_idcs.begin(), lb_idcs.end(), ub_idcs.begin(), ub_idcs.end(),
                   std::back_inserter(ineq_idcs));
    auto n_ineq = static_cast<index_t>(ineq_idcs.size());
    Points ineq_points = points_(ineq_idcs, Eigen::all);

    Solver solver(model_, bbox_);
    Evaluator res_eval(model_, bbox_, precision::kPrecise);

    common::valuesd weights = common::valuesd::Zero(n_points_ + n_poly_basis_);
    auto centers = eq_idcs;
    common::valuesd center_weights;
    std::set<index_t> active_lb_idcs;
    std::set<index_t> active_ub_idcs;

    while (true) {
      std::cout << "Active lower bounds: " << active_lb_idcs.size() << " / " << lb_idcs.size()
                << std::endl;
      std::cout << "Active upper bounds: " << active_ub_idcs.size() << " / " << ub_idcs.size()
                << std::endl;

      // Update centers (equality points + active inequality points).

      centers.resize(n_eq);
      std::set_union(active_lb_idcs.begin(), active_lb_idcs.end(), active_ub_idcs.begin(),
                     active_ub_idcs.end(), std::back_inserter(centers));

      // Fit and evaluate residuals at all inequality points.

      common::valuesd values_fit;
      if (!centers.empty()) {
        auto n_centers = static_cast<index_t>(centers.size());

        Points center_points = points_(centers, Eigen::all);

        common::valuesd center_values = values(centers, Eigen::all);
        for (index_t i = n_eq; i < n_centers; i++) {
          auto idx = centers.at(i);
          center_values(i) = active_lb_idcs.contains(idx) ? values_lb(idx) : values_ub(idx);
        }

        center_weights = weights(centers, Eigen::all);
        center_weights.conservativeResize(n_centers + n_poly_basis_);
        center_weights.tail(n_poly_basis_) = weights.tail(n_poly_basis_);

        solver.set_points(center_points);
        center_weights = solver.solve(center_values, absolute_tolerance, max_iter, center_weights);

        for (index_t i = 0; i < n_centers; i++) {
          auto idx = centers.at(i);
          weights(idx) = center_weights(i);
        }
        weights.tail(n_poly_basis_) = center_weights.tail(n_poly_basis_);

        res_eval.set_source_points(center_points, Points(0, kDim));
        res_eval.set_weights(center_weights);
        values_fit = res_eval.evaluate(ineq_points);
      } else {
        center_weights = common::valuesd::Zero(n_poly_basis_);
        values_fit = common::valuesd::Zero(n_ineq);
      }

      // Incorporate inactive inequality points with large residuals.

      auto indices = common::complementary_indices(centers, n_points_);
      auto n_indices = static_cast<index_t>(indices.size());
      common::valuesd residuals = common::valuesd::Zero(n_indices);
      for (index_t i = 0; i < n_indices; i++) {
        auto idx = indices.at(i);
        auto lb = values_lb(idx);
        auto ub = values_ub(idx);
        auto lb_res = std::isnan(lb) ? 0.0 : std::max(lb - values_fit(i), 0.0);
        auto ub_res = std::isnan(ub) ? 0.0 : std::max(values_fit(i) - ub, 0.0);
        residuals(i) = std::max(lb_res, ub_res);
      }
      common::zip_sort(indices.begin(), indices.end(), residuals.begin(), residuals.end(),
                       [](const auto& a, const auto& b) { return a.second > b.second; });
      point_cloud::distance_filter filter(points_, filtering_distance, indices);
      std::unordered_set<index_t> filtered_indices(filter.filtered_indices().begin(),
                                                   filter.filtered_indices().end());

      // Update the active set.

      auto active_set_changed = false;
      for (index_t i = 0; i < n_ineq; i++) {
        auto idx = ineq_idcs.at(i);

        if (!std::isnan(values_lb(idx))) {
          if (active_lb_idcs.contains(idx)) {
            if (weights(idx) <= 0.0) {
              active_lb_idcs.erase(idx);
              active_set_changed = true;
            }
          } else {
            if (values_fit(i) < values_lb(idx) - absolute_tolerance) {
              if (filtered_indices.contains(idx)) {
                active_lb_idcs.insert(idx);
                weights(idx) = 0.0;
              }
              active_set_changed = true;
            }
          }
        }
        if (!std::isnan(values_ub(idx))) {
          if (active_ub_idcs.contains(idx)) {
            if (weights(idx) >= 0.0) {
              active_ub_idcs.erase(idx);
              active_set_changed = true;
            }
          } else {
            if (values_fit(i) > values_ub(idx) + absolute_tolerance) {
              if (filtered_indices.contains(idx)) {
                active_ub_idcs.insert(idx);
                weights(idx) = 0.0;
              }
              active_set_changed = true;
            }
          }
        }
      }

      if (!active_set_changed) {
        break;
      }

      filtering_distance *= 0.5;
    }

    return {std::move(centers), std::move(center_weights)};
  }

 private:
  template <class Predicate>
  static std::vector<index_t> arg_where(const common::valuesd& v, Predicate predicate) {
    std::vector<index_t> idcs;

    for (index_t i = 0; i < v.rows(); i++) {
      if (predicate(v(i))) {
        idcs.push_back(i);
      }
    }

    return idcs;
  }

  const Model& model_;
  const Points& points_;

  const index_t n_points_;
  const index_t n_poly_basis_;

  const Bbox bbox_;
};

}  // namespace polatory::interpolation
