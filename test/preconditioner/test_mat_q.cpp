#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/polynomial/monomial_basis.hpp>
#include <polatory/polynomial/polynomial_basis_base.hpp>
#include <polatory/preconditioner/mat_q.hpp>
#include <polatory/types.hpp>
#include <vector>

using polatory::Index;
using polatory::kAll;
using polatory::MatX;
using polatory::geometry::Points;
using polatory::polynomial::MonomialBasis;
using polatory::polynomial::PolynomialBasisBase;
using polatory::preconditioner::MatQ;

namespace {

template <int kDim>
Points<kDim> stretched_points(Index n) {
  Points<kDim> points = Points<kDim>::Random(n, kDim);
  for (auto k = 0; k < kDim; k++) {
    points.col(k) *= std::pow(10.0, k);
  }
  points.array() += 1e3;
  return points;
}

template <int kDim>
void test(int degree, const Points<kDim>& points, const Points<kDim>& grad_points,
          Index expected_rank) {
  auto m = points.rows() + kDim * grad_points.rows();

  MatQ<kDim> mat_q(degree, points, grad_points);
  auto l = mat_q.rank();
  const auto& indices = mat_q.indices();

  ASSERT_EQ(expected_rank, l);

  std::vector<Index> sorted_indices(indices);
  std::ranges::sort(sorted_indices);
  std::vector<Index> iota(m);
  std::iota(iota.begin(), iota.end(), Index{0});
  EXPECT_EQ(iota, sorted_indices);

  ASSERT_EQ(l, mat_q.top().rows());
  ASSERT_EQ(m - l, mat_q.top().cols());

  MatX q(m, m - l);
  q.topRows(l) = mat_q.top();
  q.bottomRows(m - l) = MatX::Identity(m - l, m - l);

  MatX p = MonomialBasis<kDim>(degree).evaluate(points, grad_points)(indices, kAll);
  MatX ptq = p.transpose() * q;

  EXPECT_LE(ptq.template lpNorm<Eigen::Infinity>(),
            1e-12 * p.template lpNorm<Eigen::Infinity>() * q.template lpNorm<Eigen::Infinity>());
}

}  // namespace

TEST(mat_q, values) {
  for (auto degree = 0; degree <= 2; degree++) {
    test<1>(degree, stretched_points<1>(50), Points<1>(0, 1),
            PolynomialBasisBase<1>::basis_size(degree));
    test<2>(degree, stretched_points<2>(50), Points<2>(0, 2),
            PolynomialBasisBase<2>::basis_size(degree));
    test<3>(degree, stretched_points<3>(50), Points<3>(0, 3),
            PolynomialBasisBase<3>::basis_size(degree));
  }
}

TEST(mat_q, values_and_grads) {
  for (auto degree = 0; degree <= 2; degree++) {
    test<1>(degree, stretched_points<1>(30), stretched_points<1>(20),
            PolynomialBasisBase<1>::basis_size(degree));
    test<2>(degree, stretched_points<2>(30), stretched_points<2>(20),
            PolynomialBasisBase<2>::basis_size(degree));
    test<3>(degree, stretched_points<3>(30), stretched_points<3>(20),
            PolynomialBasisBase<3>::basis_size(degree));
  }
}

TEST(mat_q, grads_only) {
  for (auto degree = 0; degree <= 2; degree++) {
    test<1>(degree, Points<1>(0, 1), stretched_points<1>(20),
            PolynomialBasisBase<1>::basis_size(degree) - 1);
    test<2>(degree, Points<2>(0, 2), stretched_points<2>(20),
            PolynomialBasisBase<2>::basis_size(degree) - 1);
    test<3>(degree, Points<3>(0, 3), stretched_points<3>(20),
            PolynomialBasisBase<3>::basis_size(degree) - 1);
  }
}

TEST(mat_q, special_case) {
  test<1>(1, stretched_points<1>(1), stretched_points<1>(20), 2);
  test<2>(1, stretched_points<2>(1), stretched_points<2>(20), 3);
  test<3>(1, stretched_points<3>(1), stretched_points<3>(20), 4);
}

TEST(mat_q, coplanar) {
  Points<3> points = stretched_points<3>(50);
  points.col(2).array() = 1e3;

  test<3>(0, points, Points<3>(0, 3), 1);
  test<3>(1, points, Points<3>(0, 3), 3);
  test<3>(2, points, Points<3>(0, 3), 6);
}

TEST(mat_q, oblique_plane) {
  Points<3> points = stretched_points<3>(50);
  points.col(2) = 0.3 * points.col(0) - 0.02 * points.col(1);

  test<3>(1, points, Points<3>(0, 3), 3);
  test<3>(2, points, Points<3>(0, 3), 6);
  test<3>(2, points.topRows(20), stretched_points<3>(10), 10);
}
