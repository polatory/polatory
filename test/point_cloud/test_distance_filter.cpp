#include <gtest/gtest.h>

#include <polatory/geometry/point3d.hpp>
#include <polatory/point_cloud/distance_filter.hpp>
#include <polatory/types.hpp>
#include <vector>

using polatory::Index;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::point_cloud::DistanceFilter;

TEST(distance_filter, trivial) {
  Points3 points(9, 3);
  points << Point3(0, 0, 0), Point3(0, 0, 0), Point3(0, 0, 0), Point3(1, 0, 0), Point3(1, 0, 0),
      Point3(1, 0, 0), Point3(2, 0, 0), Point3(2, 0, 0), Point3(2, 0, 0);

  DistanceFilter filter(points);
  filter.filter(0.5);

  std::vector<Index> expected_filtered_indices{0, 3, 6};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices());
}

TEST(distance_filter, filter_distance) {
  Points3 points(7, 3);
  points << Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0), Point3(0, 0, 1), Point3(2, 0, 0),
      Point3(0, 2, 0), Point3(0, 0, 2);

  DistanceFilter filter(points);
  filter.filter(1.5);

  std::vector<Index> expected_filtered_indices{0, 4, 5, 6};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices());
}

TEST(distance_filter, non_trivial_indices) {
  Points3 points(9, 3);
  points << Point3(0, 0, 0), Point3(0, 0, 0), Point3(0, 0, 0), Point3(1, 0, 0), Point3(1, 0, 0),
      Point3(1, 0, 0), Point3(2, 0, 0), Point3(2, 0, 0), Point3(2, 0, 0);

  std::vector<Index> indices{8, 7, 6, 5, 4, 3, 2, 1};

  DistanceFilter filter(points);
  filter.filter(0.5, indices);

  std::vector<Index> expected_filtered_indices{8, 5, 2};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices());
}
