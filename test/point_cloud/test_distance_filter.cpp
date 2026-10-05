#include <gtest/gtest.h>

#include <jizai/geometry/point3d.hpp>
#include <jizai/point_cloud/distance_filter.hpp>
#include <jizai/types.hpp>
#include <vector>

using jizai::Index;
using jizai::geometry::Point3;
using jizai::geometry::Points3;
using jizai::point_cloud::DistanceFilter;

TEST(distance_filter, trivial) {
  Points3 points(9, 3);
  points << Point3(0, 0, 0), Point3(0, 0, 0), Point3(0, 0, 0), Point3(1, 0, 0), Point3(1, 0, 0),
      Point3(1, 0, 0), Point3(2, 0, 0), Point3(2, 0, 0), Point3(2, 0, 0);

  DistanceFilter filter(points);

  std::vector<Index> expected_filtered_indices{0, 3, 6};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices(0.5));
}

TEST(distance_filter, filter_distance) {
  Points3 points(7, 3);
  points << Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0), Point3(0, 0, 1), Point3(2, 0, 0),
      Point3(0, 2, 0), Point3(0, 0, 2);

  DistanceFilter filter(points);

  std::vector<Index> expected_filtered_indices{0, 4, 5, 6};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices(1.5));
}

TEST(distance_filter, non_trivial_indices) {
  Points3 points(9, 3);
  points << Point3(0, 0, 0), Point3(0, 0, 0), Point3(0, 0, 0), Point3(1, 0, 0), Point3(1, 0, 0),
      Point3(1, 0, 0), Point3(2, 0, 0), Point3(2, 0, 0), Point3(2, 0, 0);

  std::vector<Index> indices{8, 7, 6, 5, 4, 3, 2, 1};

  DistanceFilter filter(points);

  std::vector<Index> expected_filtered_indices{8, 5, 2};

  EXPECT_EQ(expected_filtered_indices, filter.filtered_indices(0.5, indices));
}
