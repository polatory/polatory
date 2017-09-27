// Copyright (c) 2016, GSI and The Polatory Authors.

#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "distribution_generator/spherical_distribution.hpp"
#include "interpolation/rbf_direct_evaluator.hpp"
#include "interpolation/rbf_evaluator.hpp"
#include "polynomial/basis_base.hpp"
#include "rbf/linear_variogram.hpp"

using namespace polatory::interpolation;
using polatory::distribution_generator::spherical_distribution;
using polatory::polynomial::basis_base;
using polatory::rbf::linear_variogram;

namespace {

template<class Evaluator>
void test_poly_degree(int poly_degree, size_t n_points, size_t n_eval_points)
{
   size_t n_polynomials = basis_base::dimension(poly_degree);
   Eigen::Vector3d center = Eigen::Vector3d::Zero();
   double radius = 1.0;//1e5;
   double absolute_tolerance = 5e-7;

   linear_variogram rbf({ 1.0, 0.0 });

   auto points = spherical_distribution(n_points, center, radius);

   Eigen::VectorXd weights = Eigen::VectorXd::Random(n_points + n_polynomials);

   rbf_direct_evaluator direct_eval(rbf, poly_degree, points);
   direct_eval.set_weights(weights);
   direct_eval.set_field_points(std::vector<Eigen::Vector3d>(points.begin(), points.begin() + n_eval_points));

   Evaluator eval(rbf, poly_degree, points);
   eval.set_weights(weights);
   eval.set_field_points(points);

   auto direct_values = direct_eval.evaluate();
   auto values = eval.evaluate();

   ASSERT_EQ(n_eval_points, direct_values.size());
   ASSERT_EQ(n_points, values.size());

   auto max_residual = (values.head(n_eval_points) - direct_values).template lpNorm<Eigen::Infinity>();
   EXPECT_LT(max_residual, absolute_tolerance);
}

} // namespace

TEST(rbf_evaluator, trivial)
{
   test_poly_degree<rbf_evaluator<>>(-1, 32768, 1024);
   test_poly_degree<rbf_evaluator<>>(0, 32768, 1024);
   test_poly_degree<rbf_evaluator<>>(1, 32768, 1024);
   test_poly_degree<rbf_evaluator<>>(2, 32768, 1024);
}
