// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <limits>

#include <ceres/ceres.h>

#include <polatory/rbf/rbf_kernel.hpp>

namespace polatory {
namespace rbf {

class covariance_function : public rbf_kernel {
public:
  using rbf_kernel::rbf_kernel;

  double nugget() const override {
    return parameters()[2];
  }

  int order_of_cpd() const override {
    return 0;
  }

  double partial_sill() const {
    return parameters()[0];
  }

  double range() const {
    return parameters()[1];
  }

  double sill() const {
    return partial_sill() + nugget();
  }

  int num_parameters() const override {
    return 3;
  }

  const double *parameter_lower_bounds() const override {
    static const double lower_bounds[]{ 0.0, 0.0, 0.0 };
    return lower_bounds;
  }

  const double *parameter_upper_bounds() const override {
    static const double upper_bounds[]{ std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity() };
    return upper_bounds;
  }
};

#define POLATORY_DEFINE_COST_FUNCTIONS(RBF, NPARAMS)                        \
struct residual {                                                           \
  residual(double h, double gamma, double weight)                           \
    : h_(h), gamma_(gamma), weight_(weight) {}                              \
                                                                            \
  template <class T>                                                        \
  bool operator()(const T *params, T *residual) const {                     \
    auto sill = params[0] + params[2];                                      \
    auto model_gamma = sill - RBF::evaluate(T(h_), params);                 \
    residual[0] = T(weight_) * (T(gamma_) - model_gamma);                   \
                                                                            \
    return true;                                                            \
  }                                                                         \
                                                                            \
private:                                                                    \
  const double h_;                                                          \
  const double gamma_;                                                      \
  const double weight_;                                                     \
};                                                                          \
                                                                            \
struct residual_over_gamma {                                                \
  residual_over_gamma(double h, double gamma, double weight)                \
    : h_(h), gamma_(gamma), weight_(weight) {}                              \
                                                                            \
  template <class T>                                                        \
  bool operator()(const T *params, T *residual) const {                     \
    auto sill = params[0] + params[2];                                      \
    auto model_gamma = sill - RBF::evaluate(T(h_), params);                 \
    residual[0] = T(weight_) * (T(gamma_) - model_gamma) / model_gamma;     \
                                                                            \
    return true;                                                            \
  }                                                                         \
                                                                            \
private:                                                                    \
  const double h_;                                                          \
  const double gamma_;                                                      \
  const double weight_;                                                     \
};                                                                          \
                                                                            \
ceres::CostFunction *cost_function(double h, double gamma, double weight = 1.0) const override { \
  return new ceres::NumericDiffCostFunction<residual, ceres::CENTRAL, 1, NPARAMS>( \
    new residual(h, gamma, weight)                                          \
  );                                                                        \
}                                                                           \
                                                                            \
ceres::CostFunction *cost_function_over_gamma(double h, double gamma, double weight = 1.0) const override { \
  return new ceres::NumericDiffCostFunction<residual_over_gamma, ceres::CENTRAL, 1, NPARAMS>( \
    new residual_over_gamma(h, gamma, weight)                               \
  );                                                                        \
}

} // namespace rbf
} // namespace polatory
