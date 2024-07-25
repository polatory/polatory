#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/types.hpp>
#include <vector>

using polatory::index_t;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::point_cloud::distance_filter;

TEST(distance_filter, trivial) {
  points3d points(9, 3);
  points << point3d(0, 0, 0), point3d(0, 0, 0), point3d(0, 0, 0), point3d(1, 0, 0),
      point3d(1, 0, 0), point3d(1, 0, 0), point3d(2, 0, 0), point3d(2, 0, 0), point3d(2, 0, 0);

  distance_filter filter(points);
  filter.filter(0.5);

  std::vector<index_t> expected_filtered_indices{0, 3, 6};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices());
}

TEST(distance_filter, filter_distance) {
  points3d points(7, 3);
  points << point3d(0, 0, 0), point3d(1, 0, 0), point3d(0, 1, 0), point3d(0, 0, 1),
      point3d(2, 0, 0), point3d(0, 2, 0), point3d(0, 0, 2);

  distance_filter filter(points);
  filter.filter(1.5);

  std::vector<index_t> expected_filtered_indices{0, 4, 5, 6};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices());
}

TEST(distance_filter, non_trivial_indices) {
  points3d points(9, 3);
  points << point3d(0, 0, 0), point3d(0, 0, 0), point3d(0, 0, 0), point3d(1, 0, 0),
      point3d(1, 0, 0), point3d(1, 0, 0), point3d(2, 0, 0), point3d(2, 0, 0), point3d(2, 0, 0);

  std::vector<index_t> indices{8, 7, 6, 5, 4, 3, 2, 1};

  distance_filter filter(points);
  filter.filter(0.5, indices);

  std::vector<index_t> expected_filtered_indices{8, 5, 2};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices());
}
