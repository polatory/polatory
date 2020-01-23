// Copyright (c) 2016, GSI and The Polatory Authors.

#include <cmath>
#include <random>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/biharmonic2d.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_quasi_spherical3.hpp>
#include <polatory/rbf/cov_quasi_spherical5.hpp>
#include <polatory/rbf/cov_quasi_spherical7.hpp>
#include <polatory/rbf/cov_quasi_spherical9.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/reference/cov_gaussian.hpp>
#include <polatory/rbf/reference/cov_spherical.hpp>
#include <polatory/rbf/reference/triharmonic3d.hpp>

#include "../random_anisotropy.hpp"

using polatory::geometry::to_linear_transformation3d;
using polatory::geometry::vector3d;
using polatory::rbf::biharmonic2d;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_quasi_spherical3;
using polatory::rbf::cov_quasi_spherical5;
using polatory::rbf::cov_quasi_spherical7;
using polatory::rbf::cov_quasi_spherical9;
using polatory::rbf::rbf_base;
using polatory::rbf::reference::cov_gaussian;
using polatory::rbf::reference::cov_spherical;
using polatory::rbf::reference::triharmonic3d;

namespace {

double hypot(double x, double y, double z) {
  return std::sqrt(x * x + y * y + z * z);
}

template <class T>
void test_clone(const std::vector<double>& params) {
  T rbf(params);
  rbf.set_anisotropy(to_linear_transformation3d(Eigen::Scaling(1.0, 2.0, 3.0)));

  auto cloned = rbf.clone();

  ASSERT_EQ(rbf.parameters(), cloned->parameters());
  ASSERT_EQ(rbf.anisotropy(), cloned->anisotropy());
  ASSERT_EQ(rbf.inverse_anisotropy(), cloned->inverse_anisotropy());
}

void test_gradient(const rbf_base& rbf) {
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
    rbf.evaluate_gradient_untransformed(&gradx, &grady, &gradz, x, y, z, r);

    // First-order central difference.

    auto r_x1 = hypot(x - h, y, z);
    auto r_y1 = hypot(x, y - h, z);
    auto r_z1 = hypot(x, y, z - h);

    auto r_x2 = hypot(x + h, y, z);
    auto r_y2 = hypot(x, y + h, z);
    auto r_z2 = hypot(x, y, z + h);

    auto gradx_approx = (rbf.evaluate_untransformed(r_x2) - rbf.evaluate_untransformed(r_x1)) / (2.0 * h);
    auto grady_approx = (rbf.evaluate_untransformed(r_y2) - rbf.evaluate_untransformed(r_y1)) / (2.0 * h);
    auto gradz_approx = (rbf.evaluate_untransformed(r_z2) - rbf.evaluate_untransformed(r_z1)) / (2.0 * h);

    EXPECT_LE(std::abs(gradx_approx - gradx), tolerance);
    EXPECT_LE(std::abs(grady_approx - grady), tolerance);
    EXPECT_LE(std::abs(gradz_approx - gradz), tolerance);
  }
}

}  // namespace

TEST(rbf, anisotropy) {
  auto v = vector3d(1.0, 1.0, 1.0);
  auto m = random_anisotropy();
  vector3d mv = m * v.transpose();

  biharmonic3d rbf({ 1.0 });

  auto cloned = rbf.clone();
  cloned->set_anisotropy(m);

  ASSERT_EQ(rbf.evaluate(mv), cloned->evaluate(v));
}

TEST(rbf, clone) {
  test_clone<biharmonic2d>({ 1.0 });
  test_clone<biharmonic3d>({ 1.0 });
  test_clone<cov_exponential>({ 1.0, 1.0 });
  test_clone<cov_quasi_spherical3>({ 1.0, 1.0 });
  test_clone<cov_quasi_spherical5>({ 1.0, 1.0 });
  test_clone<cov_quasi_spherical7>({ 1.0, 1.0 });
  test_clone<cov_quasi_spherical9>({ 1.0, 1.0 });

  test_clone<cov_gaussian>({ 1.0, 1.0 });
  test_clone<cov_spherical>({ 1.0, 1.0 });
  test_clone<triharmonic3d>({ 1.0 });
}

TEST(rbf, gradient) {
  test_gradient(biharmonic2d({ 1.0 }));
  test_gradient(biharmonic3d({ 1.0 }));
  test_gradient(cov_exponential({ 1.0, 1.0 }));
  test_gradient(cov_quasi_spherical3({ 1.0, 1.0 }));
  test_gradient(cov_quasi_spherical5({ 1.0, 1.0 }));
  test_gradient(cov_quasi_spherical7({ 1.0, 1.0 }));
  test_gradient(cov_quasi_spherical9({ 1.0, 1.0 }));

  test_gradient(cov_gaussian({ 1.0, 1.0 }));
  test_gradient(cov_spherical({ 1.0, 1.0 }));
  test_gradient(triharmonic3d({ 1.0 }));
}
