#pragma once

#include <cmath>
#include <jizai/geometry/point3d.hpp>
#include <jizai/kriging/variogram.hpp>
#include <jizai/kriging/variogram_builder.hpp>
#include <jizai/kriging/variogram_set.hpp>
#include <jizai/types.hpp>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jizai::kriging {

template <int Dim>
class VariogramCalculator {
  static constexpr int kDim = Dim;
  using Points = geometry::Points<kDim>;
  using Variogram = Variogram<kDim>;
  using VariogramBuilder = VariogramBuilder<kDim>;
  using VariogramSet = VariogramSet<kDim>;
  using Vector = geometry::Vector<kDim>;
  using Vectors = geometry::Vectors<kDim>;

 public:
  static const Vectors kIsotropicDirections;
  static const Vectors kAnisotropicDirections;

  VariogramCalculator(double lag_distance, Index num_lags)
      : lag_distance_(lag_distance), num_lags_(num_lags) {}

  std::optional<double> angle_tolerance() const { return angle_tolerance_; }

  VariogramSet calculate(const Points& points, const VecX& values) const {
    auto num_directions = directions_.rows();
    auto num_points = points.rows();
    auto lag_tolerance = lag_tolerance_.value_or(0.5 * lag_distance_);
    auto squared_cos_angle_tolerance =
        angle_tolerance_.has_value() ? std::pow(std::cos(*angle_tolerance_), 2) : 0.0;

    std::vector<VariogramBuilder> builders;
    for (Index k = 0; k < num_directions; k++) {
      builders.emplace_back(lag_distance_, lag_tolerance, num_lags_, directions_.row(k));
    }

#pragma omp parallel
    {
      std::vector<VariogramBuilder> local_builders;
      for (Index k = 0; k < num_directions; k++) {
        local_builders.emplace_back(lag_distance_, lag_tolerance, num_lags_, directions_.row(k));
      }

      VecX squared_dots;

#pragma omp for schedule(guided)
      for (Index i = 0; i < num_points - 1; i++) {
        auto point_i = points.row(i);
        auto value_i = values(i);

        for (Index j = i + 1; j < num_points; j++) {
          auto point_j = points.row(j);
          auto value_j = values(j);

          // Do not normalize the direction to avoid division for performance.
          Vector dir = point_j - point_i;
          squared_dots = (dir * directions_.transpose()).array().square();

          if (!angle_tolerance_.has_value()) {
            Index k{};
            squared_dots.maxCoeff(&k);
            local_builders.at(k).add_pair(point_i, point_j, value_i, value_j);
          } else {
            auto threshold = dir.squaredNorm() * squared_cos_angle_tolerance;
            for (Index k = 0; k < num_directions; k++) {
              if (squared_dots(k) >= threshold) {
                local_builders.at(k).add_pair(point_i, point_j, value_i, value_j);
              }
            }
          }
        }
      }

#pragma omp critical
      for (Index k = 0; k < num_directions; k++) {
        builders.at(k).merge(local_builders.at(k));
      }
    }

    std::vector<Variogram> variograms;
    for (auto& builder : builders) {
      auto variogram = builder.into_variogram();
      if (variogram.num_pairs() != 0) {
        variograms.push_back(std::move(variogram));
      }
    }

    return VariogramSet{std::move(variograms)};
  }

  const Vectors& directions() const { return directions_; }

  std::optional<double> lag_tolerance() const { return lag_tolerance_; }

  void set_angle_tolerance(std::optional<double> angle_tolerance) {
    if (angle_tolerance.has_value() && !(*angle_tolerance > 0.0)) {
      throw std::invalid_argument("angle_tolerance must be positive");
    }

    angle_tolerance_ = angle_tolerance;
  }

  void set_directions(const Vectors& directions) {
    if (directions.rows() == 0) {
      throw std::invalid_argument("directions must not be empty");
    }

    directions_ = directions.rowwise().normalized();
  }

  void set_lag_tolerance(std::optional<double> lag_tolerance) {
    if (lag_tolerance.has_value() && !(*lag_tolerance > 0.0)) {
      throw std::invalid_argument("lag_tolerance must be positive");
    }

    lag_tolerance_ = lag_tolerance;
  }

 private:
  double lag_distance_;
  Index num_lags_;
  std::optional<double> lag_tolerance_;
  Vectors directions_{kIsotropicDirections};
  std::optional<double> angle_tolerance_;
};

// Defining these constants here (inline) somehow leads to STATUS_HEAP_CORRUPTION on Windows.

template <>
const geometry::Vectors1 VariogramCalculator<1>::kIsotropicDirections;

template <>
const geometry::Vectors1 VariogramCalculator<1>::kAnisotropicDirections;

template <>
const geometry::Vectors2 VariogramCalculator<2>::kIsotropicDirections;

template <>
const geometry::Vectors2 VariogramCalculator<2>::kAnisotropicDirections;

template <>
const geometry::Vectors3 VariogramCalculator<3>::kIsotropicDirections;

template <>
const geometry::Vectors3 VariogramCalculator<3>::kAnisotropicDirections;

}  // namespace jizai::kriging
