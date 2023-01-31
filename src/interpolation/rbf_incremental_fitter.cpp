#include <algorithm>
#include <boost/range/irange.hpp>
#include <iostream>
#include <iterator>
#include <memory>
#include <polatory/common/zip_sort.hpp>
#include <polatory/fmm/fmm_tree_height.hpp>
#include <polatory/interpolation/rbf_incremental_fitter.hpp>
#include <polatory/interpolation/rbf_solver.hpp>
#include <random>

namespace polatory::interpolation {

rbf_incremental_fitter::rbf_incremental_fitter(const model& model, const geometry::points3d& points)
    : model_(model),
      points_(points),
      n_points_(points.rows()),
      n_poly_basis_(model.poly_basis_size()),
      bbox_(geometry::bbox3d::from_points(points)) {}

std::pair<std::vector<index_t>, common::valuesd> rbf_incremental_fitter::fit(
    const common::valuesd& values, double absolute_tolerance) const {
  auto centers = initial_indices();
  auto n_centers = static_cast<index_t>(centers.size());
  common::valuesd center_weights = common::valuesd::Zero(n_centers + n_poly_basis_);

  std::unique_ptr<rbf_solver> solver;
  std::unique_ptr<rbf_evaluator<>> res_eval;
  auto last_tree_height = 0;

  while (true) {
    std::cout << "Number of RBF centers: " << n_centers << " / " << n_points_ << std::endl;

    geometry::points3d center_points = points_(centers, Eigen::all);
    auto tree_height = fmm::fmm_tree_height(n_centers);

    if (tree_height != last_tree_height) {
      solver = std::make_unique<rbf_solver>(model_, tree_height, bbox_);
      res_eval = std::make_unique<rbf_evaluator<>>(model_, tree_height, bbox_);
      last_tree_height = tree_height;
    }

    solver->set_points(center_points);
    center_weights = solver->solve(values(centers, Eigen::all), absolute_tolerance, center_weights);

    if (n_centers == n_points_) {
      break;
    }

    // Evaluate residuals at remaining points.

    auto c_centers = complement_indices(centers);
    geometry::points3d c_center_points = points_(c_centers, Eigen::all);

    res_eval->set_source_points(center_points);
    res_eval->set_weights(center_weights);

    auto c_values_fit = res_eval->evaluate(c_center_points);
    common::valuesd c_values = values(c_centers, Eigen::all);
    std::vector<double> c_residuals(c_centers.size());
    common::valuesd::Map(c_residuals.data(), static_cast<index_t>(c_centers.size())) =
        (c_values_fit - c_values).cwiseAbs();

    // Sort remaining points by their residuals.

    common::zip_sort(c_centers.begin(), c_centers.end(), c_residuals.begin(), c_residuals.end(),
                     [](const auto& a, const auto& b) { return a.second < b.second; });

    // Count points with residuals larger than absolute_tolerance.

    auto it = std::lower_bound(c_residuals.begin(), c_residuals.end(), absolute_tolerance);
    auto n_points_need_fitting = static_cast<index_t>(std::distance(it, c_residuals.end()));
    std::cout << "Number of points to fit: " << n_points_need_fitting << std::endl;

    if (n_points_need_fitting == 0) {
      break;
    }

    // Append points with the largest residuals.

    auto n_last_centers = n_centers;
    auto n_centers_to_add =
        std::min(n_points_need_fitting,
                 std::max(index_t{max_n_points_to_add},
                          static_cast<index_t>(point_adoption_ratio *
                                               static_cast<double>(n_points_need_fitting))));

    centers.insert(centers.end(), c_centers.end() - n_centers_to_add, c_centers.end());
    n_centers = static_cast<index_t>(centers.size());

    auto last_center_weights = center_weights;
    center_weights = common::valuesd::Zero(n_centers + n_poly_basis_);
    center_weights.head(n_last_centers) = last_center_weights.head(n_last_centers);
    center_weights.tail(n_poly_basis_) = last_center_weights.tail(n_poly_basis_);
  }

  return {std::move(centers), std::move(center_weights)};
}

std::vector<index_t> rbf_incremental_fitter::initial_indices() const {
  std::vector<index_t> idcs;

  idcs.resize(n_points_);
  std::iota(idcs.begin(), idcs.end(), index_t{0});

  if (n_points_ >= min_n_points_for_incremental_fitting) {
    // TODO(mizuno): Use std::sample or a data-aware sampling method.

    auto n_initial_points =
        static_cast<index_t>(initial_points_ratio * static_cast<double>(n_points_));

    std::random_device rd;
    std::mt19937 gen(rd());

    std::shuffle(idcs.begin(), idcs.end(), gen);
    idcs.resize(n_initial_points);
    idcs.shrink_to_fit();
  }

  return idcs;
}

std::vector<index_t> rbf_incremental_fitter::complement_indices(
    const std::vector<index_t>& indices) const {
  std::vector<index_t> c_idcs(n_points_ - indices.size());

  auto universe = boost::irange<index_t>(index_t{0}, n_points_);
  auto idcs = indices;
  std::sort(idcs.begin(), idcs.end());
  std::set_difference(universe.begin(), universe.end(), idcs.begin(), idcs.end(), c_idcs.begin());

  return c_idcs;
}

}  // namespace polatory::interpolation
