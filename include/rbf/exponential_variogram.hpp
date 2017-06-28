// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <limits>

#include "variogram.hpp"

namespace polatory {
namespace rbf {

struct exponential_variogram : variogram {
   using variogram::variogram;

   static double evaluate(double r, const double *params)
   {
      auto sill = params[0];
      auto range = params[1];
      auto nugget = params[2];

      return nugget + (sill - nugget) * (1.0 - std::exp(-r / range));
   }

   double evaluate(double r) const override
   {
      return evaluate(r, parameters());
   }

   void evaluate_gradient(
      double& gradx, double& grady, double& gradz,
      double x, double y, double z, double r) const override
   {
      auto sill = parameters()[0];
      auto range = parameters()[1];
      auto nugget = parameters()[2];

      auto c = (sill - nugget) * std::exp(-r / range) / (range * r);
      gradx = c * x;
      grady = c * y;
      gradz = c * z;
   }

   double nugget() const override
   {
      return parameters()[2];
   }

   const double *parameter_lower_bounds() const override
   {
      static const double lower_bounds[]{ 0.0, 0.0, 0.0 };
      return lower_bounds;
   }

   const double *parameter_upper_bounds() const override
   {
      static const double upper_bounds[]{ std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
      return upper_bounds;
   }

   int num_parameters() const override
   {
      return 3;
   }

   DECLARE_COST_FUNCTIONS(exponential_variogram)
};

DEFINE_COST_FUNCTIONS(exponential_variogram, 3)

} // namespace rbf
} // namespace polatory
