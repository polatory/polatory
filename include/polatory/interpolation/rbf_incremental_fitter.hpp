#pragma once

#include <algorithm>
#include <boost/range/irange.hpp>
#include <iostream>
#include <iterator>
#include <polatory/common/zip_sort.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/types.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::interpolation {

template <class Model>
class rbf_incremental_fitter {
  static constexpr int kDim = Model::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Solver = rbf_solver<Model>;
  using Evaluator = rbf_evaluator<Model>;

 public:
  rbf_incremental_fitter(const Model& model, const Points& points)
      : model_(model),
        points_(points),
        n_points_(points.rows()),
        n_poly_basis_(model.poly_basis_size()),
        bbox_(Bbox::from_points(points)) {}

  std::pair<std::vector<index_t>, common::valuesd> fit(const common::valuesd& values,
                                                       double absolute_tolerance,
                                                       int max_iter) const {
    auto filtering_distance = bbox_.width().mean() / 4.0;

    auto centers = point_cloud::distance_filter(points_, filtering_distance).filtered_indices();
    auto n_centers = static_cast<index_t>(centers.size());
    common::valuesd center_weights = common::valuesd::Zero(n_centers + n_poly_basis_);

    Solver solver(model_, bbox_);
    Evaluator res_eval(model_, bbox_, precision::kPrecise);

    while (true) {
      std::cout << "Number of RBF centers: " << n_centers << " / " << n_points_ << std::endl;

      Points center_points = points_(centers, Eigen::all);

      solver.set_points(center_points);
      center_weights =
          solver.solve(values(centers, Eigen::all), absolute_tolerance, max_iter, center_weights);

      if (n_centers == n_points_) {
        break;
      }

      // Evaluate residuals at remaining points.

      auto c_centers = complementary_indices(centers);
      Points c_center_points = points_(c_centers, Eigen::all);

      res_eval.set_source_points(center_points, Points(0, kDim));
      res_eval.set_weights(center_weights);

      auto c_values_fit = res_eval.evaluate(c_center_points);
      common::valuesd c_values = values(c_centers, Eigen::all);
      std::vector<double> c_residuals(c_centers.size());
      common::valuesd::Map(c_residuals.data(), static_cast<index_t>(c_centers.size())) =
          (c_values_fit - c_values).cwiseAbs();

      // Sort remaining points by their residuals.

      common::zip_sort(c_centers.begin(), c_centers.end(), c_residuals.begin(), c_residuals.end(),
                       [](const auto& a, const auto& b) { return a.second < b.second; });

      // Count points with residuals larger than absolute_tolerance.

      auto lb = std::lower_bound(c_residuals.begin(), c_residuals.end(), absolute_tolerance);
      auto n_points_need_fitting = static_cast<index_t>(std::distance(lb, c_residuals.end()));
      std::cout << "Number of points to fit: " << n_points_need_fitting << std::endl;

      if (n_points_need_fitting == 0) {
        break;
      }

      // Append points with the largest residuals.

      auto n_last_centers = n_centers;

      std::vector<index_t> indices(centers);
      std::copy(c_centers.rbegin(), c_centers.rend(), std::back_inserter(indices));
      point_cloud::distance_filter filter(points_, filtering_distance, indices);
      std::unordered_set<index_t> filtered_indices(filter.filtered_indices().begin(),
                                                   filter.filtered_indices().end());

      for (auto it = c_centers.rbegin(); it != c_centers.rbegin() + n_points_need_fitting; ++it) {
        if (filtered_indices.contains(*it)) {
          centers.push_back(*it);
        }
      }

      n_centers = static_cast<index_t>(centers.size());

      auto last_center_weights = center_weights;
      center_weights = common::valuesd::Zero(n_centers + n_poly_basis_);
      center_weights.head(n_last_centers) = last_center_weights.head(n_last_centers);
      center_weights.tail(n_poly_basis_) = last_center_weights.tail(n_poly_basis_);

      filtering_distance *= 0.5;
    }

    return {std::move(centers), std::move(center_weights)};
  }

 private:
  std::vector<index_t> complementary_indices(const std::vector<index_t>& indices) const {
    std::vector<index_t> c_idcs(n_points_ - indices.size());

    auto universe = boost::irange<index_t>(index_t{0}, n_points_);
    auto idcs = indices;
    std::sort(idcs.begin(), idcs.end());
    std::set_difference(universe.begin(), universe.end(), idcs.begin(), idcs.end(), c_idcs.begin());

    return c_idcs;
  }

  const Model& model_;
  const Points& points_;

  const index_t n_points_;
  const index_t n_poly_basis_;

  const Bbox bbox_;
};

}  // namespace polatory::interpolation
