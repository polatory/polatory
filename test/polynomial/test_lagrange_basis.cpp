#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/types.hpp>

using polatory::index_t;
using polatory::matrixd;
using polatory::geometry::cuboid3d;
using polatory::point_cloud::random_points;
using polatory::polynomial::lagrange_basis;

namespace {

template <int kDim>
void test_degree(int degree) {
  auto n_points = lagrange_basis<kDim>::basis_size(degree);

  // A constant seed is used as this test occasionally fails.
  auto points = random_points(cuboid3d(), n_points, 0);

  lagrange_basis<kDim> basis(degree, points);
  auto p = basis.evaluate(points);

  EXPECT_EQ(n_points, p.rows());
  EXPECT_EQ(basis.basis_size(), p.cols());

  matrixd diff = matrixd::Identity(n_points, n_points) - p;

  EXPECT_LT(diff.lpNorm<Eigen::Infinity>(), 1e-12);
}

}  // namespace

TEST(lagrange_basis, trivial) {
  for (auto deg = 0; deg <= 2; deg++) {
    test_degree<1>(deg);
    test_degree<2>(deg);
    test_degree<3>(deg);
  }
}
