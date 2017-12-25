// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <functional>
#include <limits>
#include <vector>

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

  size_t num_parameters() const override {
    return 3;
  }

  const std::vector<double>& parameter_lower_bounds() const override {
    static const std::vector<double> lower_bounds{ 0.0, 0.0, 0.0 };
    return lower_bounds;
  }

  const std::vector<double>& parameter_upper_bounds() const override {
    static const std::vector<double> upper_bounds{ std::numeric_limits<double>::infinity(),
                                                   std::numeric_limits<double>::infinity(),
                                                   std::numeric_limits<double>::infinity() };
    return upper_bounds;
  }
};

#define POLATORY_DEFINE_COST_FUNCTION(RBF, NPARAMS)                           \
struct residual {                                                             \
  residual(size_t n, double h, double gamma, weight_function weight)          \
    : n_(n)                                                                   \
    , h_(h)                                                                   \
    , gamma_(gamma)                                                           \
    , weight_(weight) {}                                                      \
                                                                              \
  template <class T>                                                          \
  bool operator()(const T *params, T *residual) const {                       \
    auto sill = params[0] + params[2];                                        \
    auto model_gamma = sill - RBF::evaluate(h_, params);                      \
    residual[0] = weight_(n_, h_, model_gamma) * (gamma_ - model_gamma);      \
                                                                              \
    return true;                                                              \
  }                                                                           \
                                                                              \
private:                                                                      \
  const size_t n_;                                                            \
  const double h_;                                                            \
  const double gamma_;                                                        \
  const weight_function weight_;                                              \
};                                                                            \
                                                                              \
ceres::CostFunction *cost_function(size_t n, double h, double gamma, weight_function weight) const override { \
  return new ceres::NumericDiffCostFunction<residual, ceres::CENTRAL, 1, NPARAMS>( \
    new residual(n, h, gamma, weight)                                         \
  );                                                                          \
}

}  // namespace rbf
}  // namespace polatory
