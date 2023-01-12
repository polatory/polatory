#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/biharmonic2d.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/reference/cov_gaussian.hpp>
#include <polatory/rbf/reference/cov_spherical.hpp>
#include <polatory/rbf/reference/triharmonic3d.hpp>

#include "../random_anisotropy.hpp"

using polatory::geometry::to_linear_transformation3d;
using polatory::geometry::transform_vector;
using polatory::geometry::vector3d;
using polatory::rbf::biharmonic2d;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_spheroidal3;
using polatory::rbf::cov_spheroidal5;
using polatory::rbf::cov_spheroidal7;
using polatory::rbf::cov_spheroidal9;
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
  rbf.set_anisotropy(random_anisotropy());

  auto cloned = rbf.clone();

  ASSERT_EQ(rbf.anisotropy(), cloned->anisotropy());
  ASSERT_EQ(rbf.parameters(), cloned->parameters());
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
  auto a = random_anisotropy();
  vector3d v({ 1.0, 1.0, 1.0 });

  biharmonic3d rbf({ 1.0 });

  auto cloned = rbf.clone();
  cloned->set_anisotropy(a);

  ASSERT_EQ(rbf.evaluate(transform_vector(a, v)), cloned->evaluate(v));
}

TEST(rbf, clone) {
  test_clone<biharmonic2d>({ 1.0 });
  test_clone<biharmonic3d>({ 1.0 });
  test_clone<cov_exponential>({ 1.0, 1.0 });
  test_clone<cov_spheroidal3>({ 1.0, 1.0 });
  test_clone<cov_spheroidal5>({ 1.0, 1.0 });
  test_clone<cov_spheroidal7>({ 1.0, 1.0 });
  test_clone<cov_spheroidal9>({ 1.0, 1.0 });

  test_clone<cov_gaussian>({ 1.0, 1.0 });
  test_clone<cov_spherical>({ 1.0, 1.0 });
  test_clone<triharmonic3d>({ 1.0 });
}

TEST(rbf, gradient) {
  test_gradient(biharmonic2d({ 1.0 }));
  test_gradient(biharmonic3d({ 1.0 }));
  test_gradient(cov_exponential({ 1.0, 1.0 }));
  test_gradient(cov_spheroidal3({ 1.0, 1.0 }));
  test_gradient(cov_spheroidal5({ 1.0, 1.0 }));
  test_gradient(cov_spheroidal7({ 1.0, 1.0 }));
  test_gradient(cov_spheroidal9({ 1.0, 1.0 }));

  test_gradient(cov_gaussian({ 1.0, 1.0 }));
  test_gradient(cov_spherical({ 1.0, 1.0 }));
  test_gradient(triharmonic3d({ 1.0 }));
}
