// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include "../driver/interpolant.hpp"
#include "field_function.hpp"

namespace polatory {
namespace isosurface {

struct rbf_field_function : field_function {
   rbf_field_function(const driver::interpolant& interpolant)
      : interpolant_(interpolant)
   {
   }

   Eigen::VectorXd operator()(const std::vector<Eigen::Vector3d>& points) const override
   {
      return interpolant_.evaluate_points(points);
   }

private:
   const driver::interpolant& interpolant_;
};

} // namespace isosurface
} // namespace polatory
