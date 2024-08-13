#include <gtest/gtest.h>

#include <Eigen/Core>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polynomial/lagrange_basis.hpp>
#include <polatory/types.hpp>

using polatory::Index;
using polatory::MatX;
using polatory::geometry::Cuboid3;
using polatory::point_cloud::random_points;
using polatory::polynomial::LagrangeBasis;

namespace {

template <int kDim>
void test_degree(int degree) {
  auto n_points = LagrangeBasis<kDim>::basis_size(degree);

  // A constant seed is used as this test occasionally fails.
  auto points = random_points(Cuboid3(), n_points, 0);

  LagrangeBasis<kDim> basis(degree, points);
  auto p = basis.evaluate(points);

  EXPECT_EQ(n_points, p.rows());
  EXPECT_EQ(basis.basis_size(), p.cols());

  MatX diff = MatX::Identity(n_points, n_points) - p;

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
