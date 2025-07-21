#include <gtest/gtest.h>

#include <cmath>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/cov_cubic.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_gaussian.hpp>
#include <polatory/rbf/cov_generalized_cauchy3.hpp>
#include <polatory/rbf/cov_generalized_cauchy5.hpp>
#include <polatory/rbf/cov_generalized_cauchy7.hpp>
#include <polatory/rbf/cov_generalized_cauchy9.hpp>
#include <polatory/rbf/cov_spherical.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/polyharmonic_even.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <random>

#include "../utility.hpp"

using polatory::Mat3;
using polatory::geometry::transform_vector;
using polatory::geometry::Vector3;
using polatory::rbf::internal::Biharmonic2D;
using polatory::rbf::internal::Biharmonic3D;
using polatory::rbf::internal::CovCubic;
using polatory::rbf::internal::CovExponential;
using polatory::rbf::internal::CovGaussian;
using polatory::rbf::internal::CovGeneralizedCauchy3;
using polatory::rbf::internal::CovGeneralizedCauchy5;
using polatory::rbf::internal::CovGeneralizedCauchy7;
using polatory::rbf::internal::CovGeneralizedCauchy9;
using polatory::rbf::internal::CovSpherical;
using polatory::rbf::internal::CovSpheroidal3;
using polatory::rbf::internal::CovSpheroidal5;
using polatory::rbf::internal::CovSpheroidal7;
using polatory::rbf::internal::CovSpheroidal9;
using polatory::rbf::internal::RbfBase;
using polatory::rbf::internal::Triharmonic2D;
using polatory::rbf::internal::Triharmonic3D;

namespace {

Vector3 gradient_approx(const RbfBase<3>& rbf, const Vector3& v, double h) {
  // First-order central difference.

  auto xm = Vector3{v(0) - h, v(1), v(2)};
  auto ym = Vector3{v(0), v(1) - h, v(2)};
  auto zm = Vector3{v(0), v(1), v(2) - h};

  auto xp = Vector3{v(0) + h, v(1), v(2)};
  auto yp = Vector3{v(0), v(1) + h, v(2)};
  auto zp = Vector3{v(0), v(1), v(2) + h};

  return Vector3{rbf.evaluate_isotropic(xp) - rbf.evaluate_isotropic(xm),
                 rbf.evaluate_isotropic(yp) - rbf.evaluate_isotropic(ym),
                 rbf.evaluate_isotropic(zp) - rbf.evaluate_isotropic(zm)} /
         (2.0 * h);
}

Mat3 hessian_approx(const RbfBase<3>& rbf, const Vector3& v, double h) {
  auto xm = Vector3{v(0) - h, v(1), v(2)};
  auto ym = Vector3{v(0), v(1) - h, v(2)};
  auto zm = Vector3{v(0), v(1), v(2) - h};

  auto xp = Vector3{v(0) + h, v(1), v(2)};
  auto yp = Vector3{v(0), v(1) + h, v(2)};
  auto zp = Vector3{v(0), v(1), v(2) + h};

  Mat3 m;
  m << rbf.evaluate_gradient_isotropic(xp) - rbf.evaluate_gradient_isotropic(xm),
      rbf.evaluate_gradient_isotropic(yp) - rbf.evaluate_gradient_isotropic(ym),
      rbf.evaluate_gradient_isotropic(zp) - rbf.evaluate_gradient_isotropic(zm);
  m /= 2.0 * h;

  return m;
}

void test_gradient(const RbfBase<3>& rbf) {
  const auto h = 1e-8;
  const auto tolerance = 1e-4;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(h, 2.0);

  for (int k = 0; k < 100; k++) {
    auto x = dist(gen);
    auto y = dist(gen);
    auto z = dist(gen);
    Vector3 xx{x, y, z};

    auto grad = rbf.evaluate_gradient_isotropic(xx);
    auto grad_approx = gradient_approx(rbf, xx, h);

    EXPECT_LE((grad_approx - grad).norm(), tolerance);
  }
}

void test_hessian(const RbfBase<3>& rbf) {
  const auto h = 1e-8;
  const auto tolerance = 1e-4;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(h, 2.0);

  for (int k = 0; k < 100; k++) {
    auto x = dist(gen);
    auto y = dist(gen);
    auto z = dist(gen);
    Vector3 xx{x, y, z};

    auto hess = rbf.evaluate_hessian_isotropic(xx);
    auto hess_approx = hessian_approx(rbf, xx, h);

    EXPECT_LE((hess_approx - hess).norm(), tolerance);
  }
}

}  // namespace

TEST(rbf, anisotropy) {
  auto a = random_anisotropy<3>();
  Vector3 v({1.0, 1.0, 1.0});

  Biharmonic3D<3> rbf_iso({1.0});

  Biharmonic3D<3> rbf_aniso({1.0});
  rbf_aniso.set_anisotropy(a);

  ASSERT_EQ(rbf_iso.evaluate(transform_vector<3>(a, v)), rbf_aniso.evaluate(v));
}

TEST(rbf, gradient) {
  test_gradient(Biharmonic2D<3>({1.1, 0.1}));
  test_gradient(Biharmonic3D<3>({1.1, 0.1}));
  test_gradient(CovCubic<3>({1.1, 0.9}));
  test_gradient(CovExponential<3>({1.1, 0.9}));
  test_gradient(CovGaussian<3>({1.1, 0.9}));
  test_gradient(CovGeneralizedCauchy3<3>({1.1, 0.9}));
  test_gradient(CovGeneralizedCauchy5<3>({1.1, 0.9}));
  test_gradient(CovGeneralizedCauchy7<3>({1.1, 0.9}));
  test_gradient(CovGeneralizedCauchy9<3>({1.1, 0.9}));
  test_gradient(CovSpherical<3>({1.1, 0.9}));
  test_gradient(CovSpheroidal3<3>({1.1, 0.9}));
  test_gradient(CovSpheroidal5<3>({1.1, 0.9}));
  test_gradient(CovSpheroidal7<3>({1.1, 0.9}));
  test_gradient(CovSpheroidal9<3>({1.1, 0.9}));
  test_gradient(Triharmonic2D<3>({1.1, 0.1}));
  test_gradient(Triharmonic3D<3>({1.1, 0.1}));
}

TEST(rbf, hessian) {
  test_hessian(Biharmonic2D<3>({1.1, 0.1}));
  test_hessian(Biharmonic3D<3>({1.1, 0.1}));
  // test_hessian(CovCubic<3>({1.1, 0.9}));
  test_hessian(CovExponential<3>({1.1, 0.9}));
  test_hessian(CovGaussian<3>({1.1, 0.9}));
  test_hessian(CovGeneralizedCauchy3<3>({1.1, 0.9}));
  test_hessian(CovGeneralizedCauchy5<3>({1.1, 0.9}));
  test_hessian(CovGeneralizedCauchy7<3>({1.1, 0.9}));
  test_hessian(CovGeneralizedCauchy9<3>({1.1, 0.9}));
  // test_hessian(CovSpherical<3>({1.1, 0.9}));
  test_hessian(CovSpheroidal3<3>({1.1, 0.9}));
  test_hessian(CovSpheroidal5<3>({1.1, 0.9}));
  test_hessian(CovSpheroidal7<3>({1.1, 0.9}));
  test_hessian(CovSpheroidal9<3>({1.1, 0.9}));
  test_hessian(Triharmonic2D<3>({1.1, 0.1}));
  test_hessian(Triharmonic3D<3>({1.1, 0.1}));
}
