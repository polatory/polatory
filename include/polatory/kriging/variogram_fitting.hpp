#pragma once

#include <algorithm>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/variogram.hpp>
#include <polatory/kriging/weight_function.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class variogram_fitting;

namespace internal {

template <int Dim>
void clamp_parameters(std::vector<double>& params, const model<Dim>& model) {
  auto lbs = model.parameter_lower_bounds();
  auto ubs = model.parameter_upper_bounds();
  for (index_t i = 0; i < model.num_parameters(); i++) {
    params.at(i) = std::clamp(params.at(i), lbs.at(i), ubs.at(i));
  }
}

template <int Dim>
bool compute_residuals(const model<Dim>& model, const variogram<Dim>& variog,
                       const weight_function& weight_fn, double* residuals) {
  for (const auto& rbf : model.rbfs()) {
    auto range = rbf.parameters().at(1);
    if (range == 0.0) {
      return false;
    }
  }

  const auto& dir = variog.direction();
  auto num_bins = variog.num_bins();
  for (index_t i = 0; i < num_bins; i++) {
    auto dist = variog.bin_distance().at(i);
    auto gamma = variog.bin_gamma().at(i);
    auto num_pairs = variog.bin_num_pairs().at(i);

    auto model_gamma = model.nugget();
    for (const auto& rbf : model.rbfs()) {
      model_gamma += rbf.evaluate(geometry::vectorNd<Dim>::Zero()) - rbf.evaluate(dist * dir);
    }

    auto weight = weight_fn(dist, model_gamma, num_pairs);
    residuals[i] = weight * (gamma - model_gamma);
  }

  return true;
}

}  // namespace internal

}  // namespace polatory::kriging

#include <polatory/kriging/variogram_fitting_1d.hpp>
#include <polatory/kriging/variogram_fitting_2d.hpp>
#include <polatory/kriging/variogram_fitting_3d.hpp>
