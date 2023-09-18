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

  return vector3d{rbf.evaluate_untransformed(xp) - rbf.evaluate_untransformed(xm),
                  rbf.evaluate_untransformed(yp) - rbf.evaluate_untransformed(ym),
                  rbf.evaluate_untransformed(zp) - rbf.evaluate_untransformed(zm)} /
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
  m << rbf.evaluate_gradient_untransformed(xp) - rbf.evaluate_gradient_untransformed(xm),
      rbf.evaluate_gradient_untransformed(yp) - rbf.evaluate_gradient_untransformed(ym),
      rbf.evaluate_gradient_untransformed(zp) - rbf.evaluate_gradient_untransformed(zm);
  m /= 2.0 * h;

  return m;
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
    vector3d xx{x, y, z};

    auto grad = rbf.evaluate_gradient_untransformed(xx);
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

    auto hess = rbf.evaluate_hessian_untransformed(xx);
    auto hess_approx = hessian_approx(rbf, xx, h);

    EXPECT_LE((hess_approx - hess).norm(), tolerance);
  }
}

}  // namespace

TEST(rbf, anisotropy) {
  auto a = random_anisotropy();
  vector3d v({1.0, 1.0, 1.0});

  biharmonic3d rbf({1.0});

  auto cloned = rbf.clone();
  cloned->set_anisotropy(a);

  ASSERT_EQ(rbf.evaluate(transform_vector(a, v)), cloned->evaluate(v));
}

TEST(rbf, clone) {
  test_clone<biharmonic2d>({1.1});
  test_clone<biharmonic3d>({1.1});
  test_clone<cov_exponential>({1.1, 0.9});
  test_clone<cov_spheroidal3>({1.1, 0.9});
  test_clone<cov_spheroidal5>({1.1, 0.9});
  test_clone<cov_spheroidal7>({1.1, 0.9});
  test_clone<cov_spheroidal9>({1.1, 0.9});
  test_clone<multiquadric1>({1.1, 0.1});

  test_clone<cov_gaussian>({1.1, 0.9});
  test_clone<cov_spherical>({1.1, 0.9});
  test_clone<triharmonic3d>({1.1});
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
