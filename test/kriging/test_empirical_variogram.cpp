#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <polatory/common/types.hpp>
#include <polatory/geometry/cuboid3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/types.hpp>

namespace fs = std::filesystem;
using polatory::index_t;
using polatory::common::valuesd;
using polatory::geometry::cuboid3d;
using polatory::geometry::point3d;
using polatory::geometry::points3d;
using polatory::kriging::empirical_variogram;
using polatory::point_cloud::random_points;

TEST(empirical_variogram, serialize) {
  const auto n_points = index_t{100};

  points3d points = random_points(cuboid3d(), n_points);
  valuesd values = valuesd::Random(n_points);

  auto filename = (fs::temp_directory_path() / "840297bb-21d8-42a2-a3b5-814d27d78e14").string();

  empirical_variogram<3> v(points, values, 0.1, 10);
  v.save(filename);

  empirical_variogram<3> v2(filename);

  ASSERT_TRUE(std::equal(v.bin_distance().begin(), v.bin_distance().end(),
                         v2.bin_distance().begin(), v2.bin_distance().end()));

  ASSERT_TRUE(std::equal(v.bin_gamma().begin(), v.bin_gamma().end(), v2.bin_gamma().begin(),
                         v2.bin_gamma().end()));

  ASSERT_TRUE(std::equal(v.bin_num_pairs().begin(), v.bin_num_pairs().end(),
                         v2.bin_num_pairs().begin(), v2.bin_num_pairs().end()));
}

TEST(empirical_variogram, trivial) {
  const auto n_points = index_t{4};

  // Tetrahedron vertices separated from each other by a distance d.
  auto d = 0.5;
  points3d points(n_points, 3);
  points << d * point3d(std::sqrt(3.0) / 3.0, 0.0, 0.0),
      d * point3d(-std::sqrt(3.0) / 6.0, 1.0 / 2.0, 0.0),
      d * point3d(-std::sqrt(3.0) / 6.0, -1.0 / 2.0, 0.0),
      d * point3d(0.0, 0.0, std::sqrt(6.0) / 3.0);

  valuesd values = valuesd::Random(n_points);
  valuesd centered = values.array() - values.mean();
  auto variance = centered.dot(centered) / static_cast<double>(n_points - 1);

  auto bin_width = 0.2;
  auto num_bins = index_t{5};

  empirical_variogram<3> variog(points, values, bin_width, num_bins);

  const auto& bin_distance = variog.bin_distance();
  EXPECT_EQ(1u, bin_distance.size());
  EXPECT_DOUBLE_EQ(d, bin_distance[0]);

  const auto& bin_gamma = variog.bin_gamma();
  EXPECT_EQ(1u, bin_gamma.size());
  EXPECT_DOUBLE_EQ(variance, bin_gamma[0]);

  const auto& bin_num_pairs = variog.bin_num_pairs();
  EXPECT_EQ(1u, bin_num_pairs.size());
  EXPECT_EQ(6u, bin_num_pairs[0]);
}

TEST(empirical_variogram, zero_points) {
  const auto n_points = index_t{0};

  points3d points(n_points, 3);
  valuesd values(n_points);

  auto bin_width = 0.2;
  auto num_bins = index_t{5};

  empirical_variogram variog(points, values, bin_width, num_bins);

  EXPECT_EQ(0u, variog.bin_distance().size());
  EXPECT_EQ(0u, variog.bin_gamma().size());
  EXPECT_EQ(0u, variog.bin_num_pairs().size());
}
