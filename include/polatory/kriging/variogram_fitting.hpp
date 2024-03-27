#pragma once

#include <algorithm>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class variogram_fitting {};

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
double compute_model_gamma(const model<Dim>& model, const geometry::vectorNd<Dim>& lag) {
  auto gamma = model.nugget();
  for (const auto& rbf : model.rbfs()) {
    gamma += rbf.evaluate(geometry::vectorNd<Dim>::Zero()) - rbf.evaluate(lag);
  }
  return gamma;
}

}  // namespace internal

}  // namespace polatory::kriging

#include <polatory/kriging/variogram_fitting_1d.hpp>
#include <polatory/kriging/variogram_fitting_2d.hpp>
#include <polatory/kriging/variogram_fitting_3d.hpp>
