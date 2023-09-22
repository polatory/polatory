#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/preconditioner/domain_divider.hpp>
#include <polatory/types.hpp>
#include <vector>

using polatory::index_t;
using polatory::geometry::sphere3d;
using polatory::point_cloud::random_points;
using polatory::preconditioner::domain_divider;

TEST(domain_divider, trivial) {
  auto n_points = index_t{10000};
  auto n_grad_points = index_t{10000};
  auto n_poly_points = index_t{10};

  auto points = random_points(sphere3d(), n_points);
  std::vector<index_t> point_idcs(n_points);
  std::iota(point_idcs.begin(), point_idcs.end(), 0);

  auto grad_points = random_points(sphere3d(), n_grad_points);
  std::vector<index_t> grad_point_idcs(n_grad_points);
  std::iota(grad_point_idcs.begin(), grad_point_idcs.end(), 0);

  std::vector<index_t> poly_point_idcs(point_idcs.begin(), point_idcs.begin() + n_poly_points);

  domain_divider divider(points, grad_points, point_idcs, grad_point_idcs, poly_point_idcs);

  std::vector<index_t> inner_points;
  std::vector<index_t> inner_grad_points;
  inner_points.reserve(n_points);
  inner_grad_points.reserve(n_grad_points);
  for (const auto& d : divider.domains()) {
    for (index_t i = 0; i < d.size(); i++) {
      if (d.inner_point.at(i)) {
        inner_points.push_back(d.point_indices.at(i));
      }
    }

    for (index_t i = 0; i < d.grad_size(); i++) {
      if (d.inner_grad_point.at(i)) {
        inner_grad_points.push_back(d.grad_point_indices.at(i));
      }
    }

    for (index_t i = 0; i < n_poly_points; i++) {
      EXPECT_EQ(poly_point_idcs.at(i), d.point_indices.at(i));
    }

    std::vector<index_t> d_point_idcs(d.point_indices);
    std::sort(d_point_idcs.begin(), d_point_idcs.end());
    EXPECT_EQ(d_point_idcs.end(), std::unique(d_point_idcs.begin(), d_point_idcs.end()));

    std::vector<index_t> d_grad_point_idcs(d.grad_point_indices);
    std::sort(d_grad_point_idcs.begin(), d_grad_point_idcs.end());
    EXPECT_EQ(d_grad_point_idcs.end(),
              std::unique(d_grad_point_idcs.begin(), d_grad_point_idcs.end()));
  }
  EXPECT_EQ(n_points, inner_points.size());
  EXPECT_EQ(n_grad_points, inner_grad_points.size());

  std::sort(inner_points.begin(), inner_points.end());
  EXPECT_EQ(inner_points.end(), std::unique(inner_points.begin(), inner_points.end()));

  std::sort(inner_grad_points.begin(), inner_grad_points.end());
  EXPECT_EQ(inner_grad_points.end(),
            std::unique(inner_grad_points.begin(), inner_grad_points.end()));

  auto coarse_ratio = 0.1;
  auto [coarse_point_idcs, coarse_grad_point_idcs] = divider.choose_coarse_points(coarse_ratio);
  EXPECT_LE(0.95 * coarse_ratio * n_points, coarse_point_idcs.size());
  EXPECT_GE(1.05 * coarse_ratio * n_points, coarse_point_idcs.size());
  EXPECT_LE(0.95 * coarse_ratio * n_grad_points, coarse_grad_point_idcs.size());
  EXPECT_GE(1.05 * coarse_ratio * n_grad_points, coarse_grad_point_idcs.size());

  for (index_t i = 0; i < n_poly_points; i++) {
    EXPECT_EQ(poly_point_idcs.at(i), coarse_point_idcs.at(i));
  }

  // TODO(mizuno): Check that at least one coarse point is chosen from each domain.
}
