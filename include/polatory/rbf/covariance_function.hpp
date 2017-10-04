// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <limits>

#include "polatory/third_party/ceres/ceres.h"
#include "rbf_base.hpp"

namespace polatory {
namespace rbf {

class covariance_function : public rbf_base {
public:
  using rbf_base::rbf_base;

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

  virtual int num_parameters() const {
    return 3;
  }

  virtual const double *parameter_lower_bounds() const {
    static const double lower_bounds[]{ 0.0, 0.0, 0.0 };
    return lower_bounds;
  }

  virtual const double *parameter_upper_bounds() const {
    static const double upper_bounds[]{ std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::infinity() };
    return upper_bounds;
  }

  virtual ceres::CostFunction *cost_function(double h, double gamma, double weight) const = 0;

  virtual ceres::CostFunction *cost_function_over_gamma(double h, double gamma, double weight) const = 0;
};

#define DECLARE_COST_FUNCTIONS(RBF)                                         \
ceres::CostFunction *cost_function(double h, double gamma, double weight = 1.0) const override; \
ceres::CostFunction *cost_function_over_gamma(double h, double gamma, double weight = 1.0) const override;

#define DEFINE_COST_FUNCTIONS(RBF, NPARAMS)                                 \
namespace detail {                                                          \
                                                                            \
struct RBF##_residual {                                                     \
  RBF##_residual(double h, double gamma, double weight)                     \
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
struct RBF##_residual_over_gamma {                                          \
  RBF##_residual_over_gamma(double h, double gamma, double weight)          \
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
}                                                                           \
                                                                            \
inline                                                                      \
ceres::CostFunction *RBF::cost_function(double h, double gamma, double weight) const { \
  return new ceres::NumericDiffCostFunction<detail::RBF##_residual, ceres::CENTRAL, 1, NPARAMS>( \
    new detail::RBF##_residual(h, gamma, weight)                            \
  );                                                                        \
}                                                                           \
                                                                            \
inline                                                                      \
ceres::CostFunction *RBF::cost_function_over_gamma(double h, double gamma, double weight) const { \
  return new ceres::NumericDiffCostFunction<detail::RBF##_residual_over_gamma, ceres::CENTRAL, 1, NPARAMS>( \
    new detail::RBF##_residual_over_gamma(h, gamma, weight)                 \
  );                                                                        \
}

} // namespace rbf
} // namespace polatory
