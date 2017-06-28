// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include "field_function.hpp"
#include "../interpolation/rbf_evaluator.hpp"

namespace polatory {
namespace isosurface {

struct rbf_field_function : field_function {
   rbf_field_function(const interpolation::rbf_evaluator<>& rbf_eval)
      : rbf_eval(rbf_eval)
   {
   }

   Eigen::VectorXd operator()(const std::vector<Eigen::Vector3d>& points) const override
   {
      return rbf_eval.evaluate_points(points);
   }

private:
   const interpolation::rbf_evaluator<>& rbf_eval;
};

} // namespace isosurface
} // namespace polatory
