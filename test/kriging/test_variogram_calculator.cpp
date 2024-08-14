#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <polatory/common/io.hpp>
#include <polatory/geometry/cuboid3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/variogram_calculator.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/types.hpp>

namespace fs = std::filesystem;
using polatory::Index;
using polatory::VecX;
using polatory::geometry::Cuboid3;
using polatory::geometry::Point3;
using polatory::geometry::Points3;
using polatory::kriging::VariogramCalculator;
using polatory::kriging::VariogramSet;
using polatory::point_cloud::random_points;

TEST(variogram_calculator, serialization) {
  const auto n_points = Index{1000};

  Points3 points = random_points(Cuboid3(), n_points);
  VecX values = VecX::Random(n_points);

  auto filename = (fs::temp_directory_path() / "840297bb-21d8-42a2-a3b5-814d27d78e14").string();

  VariogramCalculator<3> calc(0.1, 10);
  calc.set_directions(VariogramCalculator<3>::kAnisotropicDirections);
  auto variog_set = calc.calculate(points, values);

  variog_set.save(filename);
  auto variog_set2 = VariogramSet<3>::load(filename);

  EXPECT_EQ(variog_set.num_variograms(), variog_set2.num_variograms());

  for (Index i = 0; i < variog_set.num_variograms(); ++i) {
    const auto& v = variog_set.variograms().at(i);
    const auto& v2 = variog_set2.variograms().at(i);

    EXPECT_EQ(v.num_bins(), v2.num_bins());
    EXPECT_TRUE(
        std::equal(v.bin_distance().begin(), v.bin_distance().end(), v2.bin_distance().begin()));
    EXPECT_TRUE(std::equal(v.bin_gamma().begin(), v.bin_gamma().end(), v2.bin_gamma().begin()));
    EXPECT_TRUE(
        std::equal(v.bin_num_pairs().begin(), v.bin_num_pairs().end(), v2.bin_num_pairs().begin()));
  }
}

TEST(variogram_calculator, trivial) {
  const auto n_points = Index{4};

  // Tetrahedron vertices separated from each other by a distance d.
  auto d = 2.0;
  Points3 points(n_points, 3);
  points << d * Point3(std::sqrt(3.0) / 3.0, 0.0, 0.0),
      d * Point3(-std::sqrt(3.0) / 6.0, 1.0 / 2.0, 0.0),
      d * Point3(-std::sqrt(3.0) / 6.0, -1.0 / 2.0, 0.0),
      d * Point3(0.0, 0.0, std::sqrt(6.0) / 3.0);

  VecX values = VecX::Random(n_points);
  VecX centered = values.array() - values.mean();
  auto variance = centered.dot(centered) / static_cast<double>(n_points - 1);

  auto lag_distance = 1.0;
  auto num_lags = Index{5};

  VariogramCalculator<3> calc(lag_distance, num_lags);
  auto variog_set = calc.calculate(points, values);
  const auto& v = variog_set.variograms().at(0);

  EXPECT_EQ(1, v.num_bins());

  EXPECT_DOUBLE_EQ(d, v.bin_distance().at(0));
  EXPECT_DOUBLE_EQ(variance, v.bin_gamma().at(0));
  EXPECT_EQ(6u, v.bin_num_pairs().at(0));
}

TEST(variogram_calculator, zero_points) {
  const auto n_points = Index{0};

  Points3 points(n_points, 3);
  VecX values(n_points);

  auto lag_distance = 0.2;
  auto num_lags = Index{5};

  VariogramCalculator<3> calc(lag_distance, num_lags);
  auto variogs_set = calc.calculate(points, values);

  EXPECT_EQ(0, variogs_set.num_variograms());
}
