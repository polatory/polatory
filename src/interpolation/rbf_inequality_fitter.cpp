// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/interpolation/rbf_inequality_fitter.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>

#include <polatory/common/eigen_utility.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/interpolation/rbf_solver.hpp>

namespace polatory {
namespace interpolation {

rbf_inequality_fitter::rbf_inequality_fitter(const model& model, const geometry::points3d& points)
  : model_(model)
  , points_(points)
  , n_points_(points.rows())
  , n_poly_basis_(model.poly_basis_size())
  , bbox_(geometry::bbox3d::from_points(points)) {
}

std::pair<std::vector<size_t>, common::valuesd>
rbf_inequality_fitter::fit(const common::valuesd& values, const common::valuesd& values_lb, const common::valuesd& values_ub,
                           double absolute_tolerance) const {
  auto not_nan = [](double d) { return !std::isnan(d); };

  auto centers = arg_where(values, not_nan);
  auto n_eq = centers.size();

  auto idcs_lb = arg_where(values_lb, not_nan);
  auto idcs_ub = arg_where(values_ub, not_nan);
  std::set<size_t> active_idcs_lb;
  std::set<size_t> active_idcs_ub;

  std::vector<size_t> ineq_idcs;
  std::set_union(idcs_lb.begin(), idcs_lb.end(),
                 idcs_ub.begin(), idcs_ub.end(),
                 std::back_inserter(ineq_idcs));
  auto ineq_points = common::take_rows(points_, ineq_idcs);

  std::unique_ptr<rbf_solver> solver;
  std::unique_ptr<rbf_evaluator<>> res_eval;
  auto last_tree_height = 0;

  common::valuesd weights = common::valuesd::Zero(n_points_ + n_poly_basis_);
  common::valuesd center_weights;

  while (true) {
    std::cout << "Active lower bounds: " << active_idcs_lb.size() << " / " << idcs_lb.size() << std::endl;
    std::cout << "Active upper bounds: " << active_idcs_ub.size() << " / " << idcs_ub.size() << std::endl;

    std::vector<size_t> active_ineq_idcs;
    std::set_union(active_idcs_lb.begin(), active_idcs_lb.end(),
                   active_idcs_ub.begin(), active_idcs_ub.end(),
                   std::back_inserter(active_ineq_idcs));

    centers.resize(n_eq);
    centers.insert(centers.end(), active_ineq_idcs.begin(), active_ineq_idcs.end());

    common::valuesd values_fit;
    if (centers.size() > 0) {
      auto tree_height = fmm::fmm_tree_height(centers.size());
      if (tree_height != last_tree_height) {
        solver = std::make_unique<rbf_solver>(model_, tree_height, bbox_);
        res_eval = std::make_unique<rbf_evaluator<>>(model_, tree_height, bbox_);
        last_tree_height = tree_height;
      }

      auto center_points = common::take_rows(points_, centers);

      auto center_values = common::take_rows(values, centers);
      for (size_t i = n_eq; i < centers.size(); i++) {
        auto idx = centers[i];
        if (active_idcs_lb.count(idx)) {
          center_values(i) = values_lb(idx);
        } else {
          center_values(i) = values_ub(idx);
        }
      }

      center_weights = common::take_rows(weights, centers);
      center_weights.conservativeResize(centers.size() + n_poly_basis_);
      center_weights.tail(n_poly_basis_) = weights.tail(n_poly_basis_);

      solver->set_points(center_points);
      center_weights = solver->solve(center_values, absolute_tolerance, center_weights);

      for (size_t i = 0; i < centers.size(); i++) {
        auto idx = centers[i];
        weights(idx) = center_weights(i);
      }
      weights.tail(n_poly_basis_) = center_weights.tail(n_poly_basis_);

      res_eval->set_source_points(center_points);
      res_eval->set_weights(center_weights);
      values_fit = res_eval->evaluate_points(ineq_points);
    } else {
      values_fit = common::valuesd::Zero(ineq_idcs.size());
    }

    size_t i = 0;
    bool active_set_changed = false;
    for (auto idx : ineq_idcs) {
      if (std::find(idcs_lb.begin(), idcs_lb.end(), idx) != idcs_lb.end()) {
        if (active_idcs_lb.count(idx)) {
          if (weights(idx) <= 0.0) {
            active_idcs_lb.erase(idx);
            active_set_changed = true;
          }
        } else {
          if (values_fit(i) < values_lb(idx) - absolute_tolerance) {
            active_idcs_lb.insert(idx);
            active_set_changed = true;
          }
        }
      }
      if (std::find(idcs_ub.begin(), idcs_ub.end(), idx) != idcs_ub.end()) {
        if (active_idcs_ub.count(idx)) {
          if (weights(idx) >= 0.0) {
            active_idcs_ub.erase(idx);
            active_set_changed = true;
          }
        } else {
          if (values_fit(i) > values_ub(idx) + absolute_tolerance) {
            active_idcs_ub.insert(idx);
            active_set_changed = true;
          }
        }
      }
      i++;
    }

    if (!active_set_changed)
      break;
  }

  return std::make_pair(std::move(centers), std::move(center_weights));
}

std::vector<size_t> rbf_inequality_fitter::arg_where(const common::valuesd& v,
                                                     std::function<bool(double)> predicate) {
  std::vector<size_t> idcs;

  for (size_t i = 0; i < v.size(); i++) {
    if (predicate(v(i)))
      idcs.push_back(i);
  }

  return idcs;
}

}  // namespace interpolation
}  // namespace polatory
