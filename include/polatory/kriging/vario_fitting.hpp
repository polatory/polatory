#pragma once

#include <ceres/autodiff_manifold.h>

#include <algorithm>
#include <numbers>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/variogram.hpp>
#include <polatory/kriging/weight_function.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <string>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class vario_fitting {};

namespace internal {

// https://github.com/ceres-solver/ceres-solver/blob/master/examples/slam/pose_graph_2d/angle_manifold.h
class AngleManifold {
 public:
  template <class T>
  bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
    *x_plus_delta = NormalizeAngle(*x + *delta);
    return true;
  }

  template <class T>
  bool Minus(const T* y, const T* x, T* y_minus_x) const {
    *y_minus_x = NormalizeAngle(*y) - NormalizeAngle(*x);
    return true;
  }

  static ceres::Manifold* Create() { return new ceres::AutoDiffManifold<AngleManifold, 1, 1>; }

 private:
  // Returns the angle wrapped to the range [-pi, pi).
  template <class T>
  static T NormalizeAngle(const T& angle) {
    T pi(std::numbers::pi);
    T two_pi(2.0 * std::numbers::pi);
    return angle - two_pi * ceres::floor((angle + pi) / two_pi);
  }
};

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

#include <polatory/kriging/vario_fitting_1d.hpp>
#include <polatory/kriging/vario_fitting_2d.hpp>
#include <polatory/kriging/vario_fitting_3d.hpp>
