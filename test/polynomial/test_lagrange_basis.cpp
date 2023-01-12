#include <Eigen/Core>
#include <gtest/gtest.h>

#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/types.hpp>

using polatory::geometry::cuboid3d;
using polatory::point_cloud::random_points;
using polatory::polynomial::lagrange_basis;
using polatory::index_t;

namespace {

void test_degree(int dimension, int degree) {
  auto n_points = lagrange_basis::basis_size(dimension, degree);

  // A constant seed is used as this test occasionally fails.
  auto points = random_points(cuboid3d(), n_points, 0);

  lagrange_basis basis(dimension, degree, points);
  auto pt = basis.evaluate(points);

  EXPECT_EQ(basis.basis_size(), pt.rows());
  EXPECT_EQ(n_points, pt.cols());

  Eigen::MatrixXd diff = Eigen::MatrixXd::Identity(n_points, n_points) - pt;

  EXPECT_LT(diff.lpNorm<Eigen::Infinity>(), 1e-12);
}

}  // namespace

TEST(lagrange_basis, trivial) {
  for (auto dim = 1; dim <= 3; dim++) {
    for (auto deg = 0; deg <= 2; deg++) {
      test_degree(dim, deg);
    }
  }
}
