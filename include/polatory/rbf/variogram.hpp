// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include "polatory/third_party/ceres/ceres.h"
#include "rbf_base.hpp"

namespace polatory {
namespace rbf {

struct variogram : rbf_base {
  using rbf_base::rbf_base;

  int definiteness() const override {
    return -1;
  }

  int order_of_definiteness() const override {
    return 1;
  }

  virtual const double *parameter_lower_bounds() const = 0;

  virtual const double *parameter_upper_bounds() const = 0;

  virtual int num_parameters() const = 0;

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
   RBF##_residual(double h, double gamma, double weight)                    \
      : h_(h), gamma_(gamma), weight_(weight) {}                            \
                                                                            \
   template<typename T>                                                     \
   bool operator()(const T *params, T *residual) const                      \
   {                                                                        \
      residual[0] = T(weight_) * (T(gamma_) - RBF::evaluate(T(h_), params)); \
                                                                            \
      return true;                                                          \
   }                                                                        \
                                                                            \
private:                                                                    \
   const double h_;                                                         \
   const double gamma_;                                                     \
   const double weight_;                                                    \
};                                                                          \
                                                                            \
struct RBF##_residual_over_gamma {                                          \
   RBF##_residual_over_gamma(double h, double gamma, double weight)         \
      : h_(h), gamma_(gamma), weight_(weight) {}                            \
                                                                            \
   template<typename T>                                                     \
   bool operator()(const T *params, T *residual) const                      \
   {                                                                        \
      auto gamma = T(RBF::evaluate(T(h_), params));                         \
      residual[0] = T(weight_) * (T(gamma_) - gamma) / gamma;               \
                                                                            \
      return true;                                                          \
   }                                                                        \
                                                                            \
private:                                                                    \
   const double h_;                                                         \
   const double gamma_;                                                     \
   const double weight_;                                                    \
};                                                                          \
                                                                            \
}                                                                           \
                                                                            \
inline                                                                      \
ceres::CostFunction *RBF::cost_function(double h, double gamma, double weight) const \
{                                                                           \
   return new ceres::NumericDiffCostFunction<detail::RBF##_residual, ceres::CENTRAL, 1, NPARAMS>( \
      new detail::RBF##_residual(h, gamma, weight)                          \
      );                                                                    \
}                                                                           \
                                                                            \
inline                                                                      \
ceres::CostFunction *RBF::cost_function_over_gamma(double h, double gamma, double weight) const \
{                                                                           \
   return new ceres::NumericDiffCostFunction<detail::RBF##_residual_over_gamma, ceres::CENTRAL, 1, NPARAMS>( \
      new detail::RBF##_residual_over_gamma(h, gamma, weight)               \
      );                                                                    \
}

} // namespace rbf
} // namespace polatory
