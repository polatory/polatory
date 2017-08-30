// Copyright (c) 2016, GSI and The Polatory Authors.

#include <iostream>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "common/vector_view.hpp"
#include "distribution_generator/spherical_distribution.hpp"
#include "interpolation/rbf_evaluator.hpp"
#include "interpolation/rbf_incremental_fitter.hpp"
#include "point_cloud/scattered_data_generator.hpp"
#include "polynomial/basis_base.hpp"
#include "rbf/linear_variogram.hpp"

using namespace polatory::interpolation;
using polatory::common::make_view;
using polatory::distribution_generator::spherical_distribution;
using polatory::point_cloud::scattered_data_generator;
using polatory::polynomial::basis_base;
using polatory::rbf::linear_variogram;

namespace {

auto test_points()
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dist(1.0 - 1e-4, 1.0 + 1e-4);

   size_t n_points = 10000;
   Eigen::Vector3d center = Eigen::Vector3d::Zero();
   double radius = 1.0;

   auto points = spherical_distribution(n_points, center, radius);
   auto normals = points;

   for (auto& p : points) {
      p *= dist(gen);
   }

   scattered_data_generator scatter_gen(points, normals, 2e-4, 1e-3);

   return std::make_pair(std::move(scatter_gen.scattered_points()), std::move(scatter_gen.scattered_values()));
}

void test_poly_degree(int poly_degree)
{
   std::vector<Eigen::Vector3d> points;
   Eigen::VectorXd values;
   std::tie(points, values) = test_points();

   size_t n_polynomials = basis_base::dimension(poly_degree);
   double absolute_tolerance = 1e-4;

   linear_variogram rbf({ 1.0, 0.0 });

   auto fitter = std::make_unique<rbf_incremental_fitter>(rbf, poly_degree, points);
   std::vector<size_t> point_indices;
   Eigen::VectorXd weights;

   std::tie(point_indices, weights) = fitter->fit(values, absolute_tolerance);
   EXPECT_EQ(weights.size(), point_indices.size() + n_polynomials);
   fitter.reset();

   rbf_evaluator<> eval(rbf, poly_degree, make_view(points, point_indices));
   eval.set_weights(weights);
   Eigen::VectorXd values_fit = eval.evaluate_points(points);

   auto max_residual = (values - values_fit).lpNorm<Eigen::Infinity>();
   std::cout << "Maximum residual:" << std::endl
      << "  " << max_residual << std::endl;

   EXPECT_LT(max_residual, absolute_tolerance);
}

} // namespace

TEST(rbf_incremental_fitter, trivial)
{
   test_poly_degree(0);
}
