// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "rbf/exponential_variogram.hpp"
#include "rbf/linear_variogram.hpp"
#include "rbf/rbf_base.hpp"
#include "rbf/spherical_variogram.hpp"

using namespace polatory::rbf;

namespace {

double hypot(double x, double y, double z)
{
   return std::sqrt(x * x + y * y + z * z);
}

void test_gradient(const rbf_base& kernel)
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dist(0.0, 2.0);

   const auto h = 1e-10;
   const auto threshold = 1e-5;
   for (int k = 0; k < 100; k++) {
      auto x = dist(gen);
      auto y = dist(gen);
      auto z = dist(gen);
      auto r = hypot(x, y, z);

      auto f = kernel.evaluate(r);

      double gradx, grady, gradz;
      kernel.evaluate_gradient(gradx, grady, gradz, x, y, z, r);

      auto r2x = hypot(x + h, y, z);
      auto r2y = hypot(x, y + h, z);
      auto r2z = hypot(x, y, z + h);

      auto gradx_approx = (kernel.evaluate(r2x) - f) / h;
      auto grady_approx = (kernel.evaluate(r2y) - f) / h;
      auto gradz_approx = (kernel.evaluate(r2z) - f) / h;

      EXPECT_LE(std::abs(gradx_approx - gradx), threshold);
      EXPECT_LE(std::abs(grady_approx - grady), threshold);
      EXPECT_LE(std::abs(gradz_approx - gradz), threshold);
   }
}

} // namespace

TEST(rbf, gradient)
{
   test_gradient(exponential_variogram({ 1.0, 1.0, 0.5 }));
   test_gradient(linear_variogram({ 1.0, 0.5 }));
   test_gradient(spherical_variogram({ 1.0, 1.0, 0.5 }));
}
