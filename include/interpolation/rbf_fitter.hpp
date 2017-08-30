// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <Eigen/Core>

#include "../rbf/rbf_base.hpp"
#include "rbf_solver.hpp"

namespace polatory {
namespace interpolation {

class rbf_fitter : rbf_solver {
public:
   using rbf_solver::set_points;

   template<typename Container>
   rbf_fitter(const rbf::rbf_base& rbf, int poly_degree,
      const Container& points)
      : rbf_solver(rbf, poly_degree, points)
   {
   }

   template<typename Derived>
   Eigen::VectorXd fit(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance) const
   {
      return solve(values, absolute_tolerance);
   }

   template<typename Derived, typename Derived2>
   Eigen::VectorXd fit(const Eigen::MatrixBase<Derived>& values, double absolute_tolerance, const Eigen::MatrixBase<Derived2>& initial_solution) const
   {
      return solve(values, absolute_tolerance, initial_solution);
   }
};

} // namespace interpolation
} // namespace polatory
