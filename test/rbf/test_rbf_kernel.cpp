// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include <polatory/rbf/biharmonic.hpp>
#include <polatory/rbf/biharmonic2d.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_quasi_spherical3.hpp>
#include <polatory/rbf/cov_quasi_spherical5.hpp>
#include <polatory/rbf/cov_quasi_spherical7.hpp>
#include <polatory/rbf/cov_quasi_spherical9.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/reference/cov_gaussian.hpp>
#include <polatory/rbf/reference/cov_spherical.hpp>
#include <polatory/rbf/reference/triharmonic.hpp>

using polatory::rbf::biharmonic;
using polatory::rbf::biharmonic2d;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_quasi_spherical3;
using polatory::rbf::cov_quasi_spherical5;
using polatory::rbf::cov_quasi_spherical7;
using polatory::rbf::cov_quasi_spherical9;
using polatory::rbf::rbf_base;
using polatory::rbf::reference::cov_gaussian;
using polatory::rbf::reference::cov_spherical;
using polatory::rbf::reference::triharmonic;

namespace {

double hypot(double x, double y, double z) {
  return std::sqrt(x * x + y * y + z * z);
}

void test_gradient(const rbf_base& kernel) {
  const auto h = 1e-8;
  const auto tolerance = 1e-5;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(h, 2.0);

  for (int k = 0; k < 100; k++) {
    auto x = dist(gen);
    auto y = dist(gen);
    auto z = dist(gen);
    auto r = hypot(x, y, z);

    double gradx, grady, gradz;
    kernel.evaluate_gradient(&gradx, &grady, &gradz, x, y, z, r);

    // First-order central difference.

    auto r_x1 = hypot(x - h, y, z);
    auto r_y1 = hypot(x, y - h, z);
    auto r_z1 = hypot(x, y, z - h);

    auto r_x2 = hypot(x + h, y, z);
    auto r_y2 = hypot(x, y + h, z);
    auto r_z2 = hypot(x, y, z + h);

    auto gradx_approx = (kernel.evaluate(r_x2) - kernel.evaluate(r_x1)) / (2.0 * h);
    auto grady_approx = (kernel.evaluate(r_y2) - kernel.evaluate(r_y1)) / (2.0 * h);
    auto gradz_approx = (kernel.evaluate(r_z2) - kernel.evaluate(r_z1)) / (2.0 * h);

    EXPECT_LE(std::abs(gradx_approx - gradx), tolerance);
    EXPECT_LE(std::abs(grady_approx - grady), tolerance);
    EXPECT_LE(std::abs(gradz_approx - gradz), tolerance);
  }
}

}  // namespace

TEST(rbf_kernel, gradient) {
  test_gradient(biharmonic({ 1.0, 0.0 }));
  test_gradient(biharmonic2d({ 1.0, 0.0 }));
  test_gradient(cov_exponential({ 1.0, 1.0, 0.0 }));
  test_gradient(cov_quasi_spherical3({ 1.0, 1.0, 0.0 }));
  test_gradient(cov_quasi_spherical5({ 1.0, 1.0, 0.0 }));
  test_gradient(cov_quasi_spherical7({ 1.0, 1.0, 0.0 }));
  test_gradient(cov_quasi_spherical9({ 1.0, 1.0, 0.0 }));

  test_gradient(cov_gaussian({ 1.0, 1.0, 0.0 }));
  test_gradient(cov_spherical({ 1.0, 1.0, 0.0 }));
  test_gradient(triharmonic({ 1.0, 0.0 }));
}
