#include <gtest/gtest.h>

#include <cmath>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/biharmonic2d.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/multiquadric1.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/reference/cov_gaussian.hpp>
#include <polatory/rbf/reference/cov_spherical.hpp>
#include <polatory/rbf/reference/triharmonic3d.hpp>
#include <random>

#include "../random_anisotropy.hpp"

using polatory::geometry::matrix3d;
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
using polatory::rbf::multiquadric1;
using polatory::rbf::rbf_base;
using polatory::rbf::reference::cov_gaussian;
using polatory::rbf::reference::cov_spherical;
using polatory::rbf::reference::triharmonic3d;

namespace {

vector3d gradient_approx(const rbf_base& rbf, const vector3d& v, double h) {
  // First-order central difference.

  auto xm = vector3d{v(0) - h, v(1), v(2)};
  auto ym = vector3d{v(0), v(1) - h, v(2)};
  auto zm = vector3d{v(0), v(1), v(2) - h};

  auto xp = vector3d{v(0) + h, v(1), v(2)};
  auto yp = vector3d{v(0), v(1) + h, v(2)};
  auto zp = vector3d{v(0), v(1), v(2) + h};

  return vector3d{rbf.evaluate_isotropic(xp) - rbf.evaluate_isotropic(xm),
                  rbf.evaluate_isotropic(yp) - rbf.evaluate_isotropic(ym),
                  rbf.evaluate_isotropic(zp) - rbf.evaluate_isotropic(zm)} /
         (2.0 * h);
}

matrix3d hessian_approx(const rbf_base& rbf, const vector3d& v, double h) {
  auto xm = vector3d{v(0) - h, v(1), v(2)};
  auto ym = vector3d{v(0), v(1) - h, v(2)};
  auto zm = vector3d{v(0), v(1), v(2) - h};

  auto xp = vector3d{v(0) + h, v(1), v(2)};
  auto yp = vector3d{v(0), v(1) + h, v(2)};
  auto zp = vector3d{v(0), v(1), v(2) + h};

  matrix3d m;
  m << rbf.evaluate_gradient_isotropic(xp) - rbf.evaluate_gradient_isotropic(xm),
      rbf.evaluate_gradient_isotropic(yp) - rbf.evaluate_gradient_isotropic(ym),
      rbf.evaluate_gradient_isotropic(zp) - rbf.evaluate_gradient_isotropic(zm);
  m /= 2.0 * h;

  return m;
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
    vector3d xx{x, y, z};

    auto grad = rbf.evaluate_gradient_isotropic(xx);
    auto grad_approx = gradient_approx(rbf, xx, h);

    EXPECT_LE((grad_approx - grad).norm(), tolerance);
  }
}

void test_hessian(const rbf_base& rbf) {
  const auto h = 1e-8;
  const auto tolerance = 1e-5;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(h, 2.0);

  for (int k = 0; k < 100; k++) {
    auto x = dist(gen);
    auto y = dist(gen);
    auto z = dist(gen);
    vector3d xx{x, y, z};

    auto hess = rbf.evaluate_hessian_isotropic(xx);
    auto hess_approx = hessian_approx(rbf, xx, h);

    EXPECT_LE((hess_approx - hess).norm(), tolerance);
  }
}

}  // namespace

TEST(rbf, anisotropy) {
  auto a = random_anisotropy();
  vector3d v({1.0, 1.0, 1.0});

  biharmonic3d rbf_iso({1.0});

  biharmonic3d rbf_aniso({1.0});
  rbf_aniso.set_anisotropy(a);

  ASSERT_EQ(rbf_iso.evaluate(transform_vector(a, v)), rbf_aniso.evaluate(v));
}

TEST(rbf, gradient) {
  test_gradient(biharmonic2d({1.1}));
  // test_gradient(biharmonic3d({1.1}));
  test_gradient(cov_exponential({1.1, 0.9}));
  test_gradient(cov_spheroidal3({1.1, 0.9}));
  test_gradient(cov_spheroidal5({1.1, 0.9}));
  test_gradient(cov_spheroidal7({1.1, 0.9}));
  test_gradient(cov_spheroidal9({1.1, 0.9}));
  test_gradient(multiquadric1({1.1, 0.1}));

  test_gradient(cov_gaussian({1.1, 0.9}));
  test_gradient(cov_spherical({1.1, 0.9}));
  test_gradient(triharmonic3d({1.1}));
}

TEST(rbf, hessian) {
  // test_hessian(biharmonic2d({1.1}));
  // test_hessian(biharmonic3d({1.1}));
  test_hessian(cov_exponential({1.1, 0.9}));
  test_hessian(cov_spheroidal3({1.1, 0.9}));
  test_hessian(cov_spheroidal5({1.1, 0.9}));
  test_hessian(cov_spheroidal7({1.1, 0.9}));
  test_hessian(cov_spheroidal9({1.1, 0.9}));
  test_hessian(multiquadric1({1.1, 0.1}));

  test_hessian(cov_gaussian({1.1, 0.9}));
  // test_hessian(cov_spherical({1.1, 0.9}));
  test_hessian(triharmonic3d({1.1}));
}
