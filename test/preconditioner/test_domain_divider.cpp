#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <format>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/preconditioner/domain_divider.hpp>
#include <polatory/types.hpp>
#include <vector>

using polatory::Index;
using polatory::geometry::Points;
using polatory::preconditioner::DomainDivider;

namespace {

template <int Dim>
void test() {
  std::cout << std::format("dim: {}", Dim) << std::endl;

  using DomainDivider = DomainDivider<Dim>;
  using Points = Points<Dim>;

  auto n_points = Index{10000};
  auto n_grad_points = Index{10000};

  auto points = Points(n_points, Dim);
  std::vector<Index> point_idcs(n_points);
  std::iota(point_idcs.begin(), point_idcs.end(), 0);

  auto grad_points = Points(n_grad_points, Dim);
  std::vector<Index> grad_point_idcs(n_grad_points);
  std::iota(grad_point_idcs.begin(), grad_point_idcs.end(), 0);

  DomainDivider divider(points, grad_points, point_idcs, grad_point_idcs);

  std::vector<Index> inner_points;
  std::vector<Index> inner_grad_points;
  inner_points.reserve(n_points);
  inner_grad_points.reserve(n_grad_points);
  for (const auto& d : divider.domains()) {
    for (Index i = 0; i < d.num_points(); i++) {
      if (d.inner_point.at(i)) {
        inner_points.push_back(d.point_indices.at(i));
      }
    }

    for (Index i = 0; i < d.num_grad_points(); i++) {
      if (d.inner_grad_point.at(i)) {
        inner_grad_points.push_back(d.grad_point_indices.at(i));
      }
    }

    std::vector<Index> d_point_idcs(d.point_indices);
    std::ranges::sort(d_point_idcs);
    EXPECT_EQ(d_point_idcs.end(), std::unique(d_point_idcs.begin(), d_point_idcs.end()));

    std::vector<Index> d_grad_point_idcs(d.grad_point_indices);
    std::ranges::sort(d_grad_point_idcs);
    EXPECT_EQ(d_grad_point_idcs.end(),
              std::unique(d_grad_point_idcs.begin(), d_grad_point_idcs.end()));
  }
  EXPECT_EQ(n_points, inner_points.size());
  EXPECT_EQ(n_grad_points, inner_grad_points.size());

  std::ranges::sort(inner_points);
  EXPECT_EQ(inner_points.end(), std::unique(inner_points.begin(), inner_points.end()));

  std::ranges::sort(inner_grad_points);
  EXPECT_EQ(inner_grad_points.end(),
            std::unique(inner_grad_points.begin(), inner_grad_points.end()));

  auto n_coarse_points = Index{1000};
  auto n_fixed_points = Index{10};
  auto n_fixed_grad_points = Index{5};
  std::vector<Index> fixed_point_idcs(point_idcs.begin(), point_idcs.begin() + n_fixed_points);
  std::vector<Index> fixed_grad_point_idcs(grad_point_idcs.begin(),
                                           grad_point_idcs.begin() + n_fixed_grad_points);
  auto [coarse_point_idcs, coarse_grad_point_idcs] =
      divider.choose_coarse_points(n_coarse_points, fixed_point_idcs, fixed_grad_point_idcs);
  auto n_fixed = n_fixed_points + Dim * n_fixed_grad_points;
  EXPECT_LE(n_coarse_points + n_fixed,
            coarse_point_idcs.size() + Dim * coarse_grad_point_idcs.size());
  EXPECT_GE(n_coarse_points + n_fixed + Dim,
            coarse_point_idcs.size() + Dim * coarse_grad_point_idcs.size());

  for (Index i = 0; i < n_fixed_points; i++) {
    EXPECT_EQ(fixed_point_idcs.at(i), coarse_point_idcs.at(i));
  }
  for (Index i = 0; i < n_fixed_grad_points; i++) {
    EXPECT_EQ(fixed_grad_point_idcs.at(i), coarse_grad_point_idcs.at(i));
  }

  std::ranges::sort(coarse_point_idcs);
  EXPECT_EQ(coarse_point_idcs.end(),
            std::unique(coarse_point_idcs.begin(), coarse_point_idcs.end()));
  std::ranges::sort(coarse_grad_point_idcs);
  EXPECT_EQ(coarse_grad_point_idcs.end(),
            std::unique(coarse_grad_point_idcs.begin(), coarse_grad_point_idcs.end()));
}

}  // namespace

TEST(domain_divider, trivial) {
  test<1>();
  test<2>();
  test<3>();
}
