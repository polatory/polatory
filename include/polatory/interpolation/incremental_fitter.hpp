#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <format>
#include <iostream>
#include <iterator>
#include <polatory/common/complementary_indices.hpp>
#include <polatory/common/macros.hpp>
#include <polatory/common/zip_sort.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/interpolation/evaluator.hpp>
#include <polatory/interpolation/solver.hpp>
#include <polatory/model.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/types.hpp>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace polatory::interpolation {

template <int Dim>
class IncrementalFitter {
  static constexpr int kDim = Dim;
  using Bbox = geometry::Bbox<kDim>;
  using Evaluator = Evaluator<kDim>;
  using Model = Model<kDim>;
  using Points = geometry::Points<kDim>;
  using Solver = Solver<kDim>;
  using Vectors = geometry::Vectors<kDim>;

 public:
  IncrementalFitter(const Model& model, const Points& points_full, const Points& grad_points_full)
      : model_(model),
        l_(model.poly_basis_size()),
        mu_full_(points_full.rows()),
        sigma_full_(grad_points_full.rows()),
        points_full_(points_full),
        grad_points_full_(grad_points_full),
        bbox_(Bbox::from_points(points_full).convex_hull(Bbox::from_points(grad_points_full))) {}

  std::tuple<std::vector<Index>, std::vector<Index>, VecX> fit(const VecX& values_full,
                                                               double tolerance,
                                                               double grad_tolerance, int max_iter,
                                                               double accuracy,
                                                               double grad_accuracy) const {
    POLATORY_ASSERT(values_full.size() == mu_full_ + kDim * sigma_full_);

    auto filtering_distance = bbox_.width().norm();
    point_cloud::DistanceFilter filter(points_full_);
    point_cloud::DistanceFilter grad_filter(grad_points_full_);

    std::vector<Index> centers;
    std::vector<Index> grad_centers;
    if (model_.poly_degree() == 1 && mu_full_ == 1 && sigma_full_ >= 1) {
      // When values_full(0) is zero, the point will never be added to centers.
      centers.push_back(0);
    }

    auto mu = static_cast<Index>(centers.size());
    auto sigma = static_cast<Index>(grad_centers.size());
    VecX weights = VecX::Zero(mu + kDim * sigma + l_);

    Solver solver(model_, bbox_, accuracy, grad_accuracy);
    Evaluator res_eval(model_, bbox_, accuracy, grad_accuracy);

    while (true) {
      if (mu_full_ > 0) {
        std::cout << std::format("Number of RBF centers: {} / {}", mu, mu_full_) << std::endl;
      }
      if (sigma_full_ > 0) {
        std::cout << std::format("Number of grad RBF centers: {} / {}", sigma, sigma_full_)
                  << std::endl;
      }

      Points points = points_full_(centers, Eigen::all);
      Points grad_points = grad_points_full_(grad_centers, Eigen::all);

      if (mu >= l_ || (model_.poly_degree() == 1 && mu == 1 && sigma >= 1)) {
        solver.set_points(points, grad_points);
        VecX values(mu + kDim * sigma);
        values << values_full.head(mu_full_)(centers),
            values_full.tail(kDim * sigma_full_)
                .template reshaped<Eigen::RowMajor>(sigma_full_, kDim)(grad_centers, Eigen::all)
                .template reshaped<Eigen::RowMajor>();
        weights = solver.solve(values, tolerance, grad_tolerance, max_iter, &weights);
      }

      if (mu == mu_full_ && sigma == sigma_full_) {
        break;
      }

      // Evaluate residuals at remaining points.

      auto c_centers = common::complementary_indices(centers, mu_full_);
      Points c_points = points_full_(c_centers, Eigen::all);

      auto c_grad_centers = common::complementary_indices(grad_centers, sigma_full_);
      Points c_grad_points = grad_points_full_(c_grad_centers, Eigen::all);

      res_eval.set_source_points(points, grad_points);
      res_eval.set_weights(weights);

      auto c_values_fit = res_eval.evaluate(c_points, c_grad_points);
      auto c_residuals = residuals(values_full, c_centers, c_values_fit);
      auto c_grad_residuals = grad_residuals(values_full, c_grad_centers, c_values_fit);

      // Sort remaining points by their residuals.

      common::zip_sort(c_centers.begin(), c_centers.end(), c_residuals.begin(),
                       [](const auto& a, const auto& b) { return a.second < b.second; });

      common::zip_sort(c_grad_centers.begin(), c_grad_centers.end(), c_grad_residuals.begin(),
                       [](const auto& a, const auto& b) { return a.second < b.second; });

      // Count points with residuals larger than tolerance.

      auto lb = std::lower_bound(c_residuals.begin(), c_residuals.end(), tolerance);
      auto n_points_need_fitting = static_cast<Index>(std::distance(lb, c_residuals.end()));
      if (mu_full_ > 0) {
        std::cout << "Number of points to fit: " << n_points_need_fitting << std::endl;
      }

      auto grad_lb =
          std::lower_bound(c_grad_residuals.begin(), c_grad_residuals.end(), grad_tolerance);
      auto n_grad_points_need_fitting =
          static_cast<Index>(std::distance(grad_lb, c_grad_residuals.end()));
      if (sigma_full_ > 0) {
        std::cout << "Number of grad points to fit: " << n_grad_points_need_fitting << std::endl;
      }

      if (n_points_need_fitting == 0 && n_grad_points_need_fitting == 0) {
        break;
      }

      // Append points with the largest residuals.

      auto last_mu = mu;
      auto last_sigma = sigma;

      std::vector<Index> indices(centers);
      std::copy(c_centers.rbegin(), c_centers.rend(), std::back_inserter(indices));
      filter.filter(filtering_distance, indices);
      std::unordered_set<Index> filtered_indices(filter.filtered_indices().begin(),
                                                 filter.filtered_indices().end());

      for (auto it = c_centers.rbegin(); it != c_centers.rbegin() + n_points_need_fitting; ++it) {
        if (filtered_indices.contains(*it)) {
          centers.push_back(*it);
        }
      }

      std::vector<Index> grad_indices(grad_centers);
      std::copy(c_grad_centers.rbegin(), c_grad_centers.rend(), std::back_inserter(grad_indices));
      grad_filter.filter(filtering_distance, grad_indices);
      std::unordered_set<Index> grad_filtered_indices(grad_filter.filtered_indices().begin(),
                                                      grad_filter.filtered_indices().end());

      for (auto it = c_grad_centers.rbegin();
           it != c_grad_centers.rbegin() + n_grad_points_need_fitting; ++it) {
        if (grad_filtered_indices.contains(*it)) {
          grad_centers.push_back(*it);
        }
      }

      mu = static_cast<Index>(centers.size());
      sigma = static_cast<Index>(grad_centers.size());

      VecX last_weights = weights;
      weights = VecX::Zero(mu + kDim * sigma + l_);
      weights.head(last_mu) = last_weights.head(last_mu);
      weights.segment(mu, kDim * last_sigma) = last_weights.segment(last_mu, kDim * last_sigma);
      weights.tail(l_) = last_weights.tail(l_);

      filtering_distance *= 0.5;
    }

    return {std::move(centers), std::move(grad_centers), std::move(weights)};
  }

  std::vector<double> residuals(const VecX& mixed_values_full, const std::vector<Index>& c_centers,
                                const VecX& c_mixed_values_fit) const {
    auto c_mu = static_cast<Index>(c_centers.size());

    VecX c_values = mixed_values_full(c_centers);
    VecX c_values_fit = c_mixed_values_fit.head(c_mu);

    std::vector<double> c_residuals(c_mu);
    Eigen::Map<VecX>(c_residuals.data(), c_mu) = (c_values_fit - c_values).cwiseAbs();

    return c_residuals;
  }

  std::vector<double> grad_residuals(const VecX& mixed_values_full,
                                     const std::vector<Index>& c_grad_centers,
                                     const VecX& c_mixed_values_fit) const {
    auto c_sigma = static_cast<Index>(c_grad_centers.size());

    Vectors c_grad_values =
        mixed_values_full.tail(kDim * sigma_full_)
            .template reshaped<Eigen::RowMajor>(sigma_full_, kDim)(c_grad_centers, Eigen::all);
    Vectors c_grad_values_fit =
        c_mixed_values_fit.tail(kDim * c_sigma).template reshaped<Eigen::RowMajor>(c_sigma, kDim);

    std::vector<double> c_grad_residuals(c_sigma);
    Eigen::Map<VecX>(c_grad_residuals.data(), c_sigma) =
        (c_grad_values_fit - c_grad_values).cwiseAbs().rowwise().maxCoeff();

    return c_grad_residuals;
  }

 private:
  const Model& model_;
  const Index l_;
  const Index mu_full_;
  const Index sigma_full_;
  const Points& points_full_;
  const Points& grad_points_full_;
  const Bbox bbox_;
};

}  // namespace polatory::interpolation
