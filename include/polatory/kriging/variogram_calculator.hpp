#pragma once

#include <cmath>
#include <numbers>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/variogram.hpp>
#include <polatory/kriging/variogram_builder.hpp>
#include <polatory/kriging/variogram_set.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class variogram_calculator {
  static constexpr int kDim = Dim;
  using Points = geometry::pointsNd<kDim>;
  using Variogram = variogram<kDim>;
  using VariogramBuilder = variogram_builder<kDim>;
  using VariogramSet = variogram_set<kDim>;
  using Vector = geometry::vectorNd<kDim>;
  using Vectors = geometry::vectorsNd<kDim>;

 public:
  // Non-constexpr for the sake of Python bindings.
  static inline const double kAutomaticAngleTolerance = -1.0;
  static inline const double kAutomaticLagTolerance = -1.0;

  static const Vectors kIsotropicDirections;
  static const Vectors kAnisotropicDirections;

  variogram_calculator(double lag_distance, index_t num_lags)
      : lag_distance_(lag_distance), num_lags_(num_lags) {}

  double angle_tolerance() const { return angle_tolerance_; }

  VariogramSet calculate(const Points& points, const vectord& values) const {
    auto num_directions = directions_.rows();
    auto num_points = points.rows();
    auto lag_tolerance =
        lag_tolerance_ == kAutomaticLagTolerance ? 0.5 * lag_distance_ : lag_tolerance_;
    auto squared_cos_angle_tolerance = angle_tolerance_ == kAutomaticAngleTolerance
                                           ? 0.0
                                           : std::pow(std::cos(angle_tolerance_), 2);

    std::vector<VariogramBuilder> builders;
    for (index_t k = 0; k < num_directions; k++) {
      builders.emplace_back(lag_distance_, lag_tolerance, num_lags_, directions_.row(k));
    }

#pragma omp parallel
    {
      std::vector<VariogramBuilder> local_builders;
      for (index_t k = 0; k < num_directions; k++) {
        local_builders.emplace_back(lag_distance_, lag_tolerance, num_lags_, directions_.row(k));
      }

      vectord squared_dots;

#pragma omp for schedule(dynamic)
      for (index_t i = 0; i < num_points - 1; i++) {
        auto point_i = points.row(i);
        auto value_i = values(i);

        for (index_t j = i + 1; j < num_points; j++) {
          auto point_j = points.row(j);
          auto value_j = values(j);

          // Do not normalize the direction to avoid division for performance.
          Vector dir = point_j - point_i;
          squared_dots = (dir * directions_.transpose()).array().square();

          if (angle_tolerance_ == kAutomaticAngleTolerance) {
            index_t k{};
            squared_dots.maxCoeff(&k);
            local_builders.at(k).add_pair(point_i, point_j, value_i, value_j);
          } else {
            auto threshold = dir.squaredNorm() * squared_cos_angle_tolerance;
            for (index_t k = 0; k < num_directions; k++) {
              if (squared_dots(k) >= threshold) {
                local_builders.at(k).add_pair(point_i, point_j, value_i, value_j);
              }
            }
          }
        }
      }

#pragma omp critical
      for (index_t k = 0; k < num_directions; k++) {
        builders.at(k).merge(local_builders.at(k));
      }
    }

    std::vector<Variogram> variograms;
    for (auto& builder : builders) {
      auto variogram = builder.into_variogram();
      if (variogram.num_pairs() != 0) {
        variograms.emplace_back(std::move(variogram));
      }
    }

    return VariogramSet{std::move(variograms)};
  }

  const Vectors& directions() const { return directions_; }

  double lag_tolerance() const { return lag_tolerance_; }

  void set_angle_tolerance(double angle_tolerance) {
    if (angle_tolerance != kAutomaticAngleTolerance && !(angle_tolerance > 0.0)) {
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

  void set_lag_tolerance(double lag_tolerance) {
    if (lag_tolerance != kAutomaticAngleTolerance && !(lag_tolerance > 0.0)) {
      throw std::invalid_argument("lag_tolerance must be positive");
    }

    lag_tolerance_ = lag_tolerance;
  }

 private:
  double lag_distance_;
  index_t num_lags_;
  double lag_tolerance_{kAutomaticLagTolerance};
  Vectors directions_{kIsotropicDirections};
  double angle_tolerance_{kAutomaticAngleTolerance};
};

// Defining these constants here (inline) somehow leads to STATUS_HEAP_CORRUPTION on Windows.

template <>
const geometry::vectors1d variogram_calculator<1>::kIsotropicDirections;

template <>
const geometry::vectors1d variogram_calculator<1>::kAnisotropicDirections;

template <>
const geometry::vectors2d variogram_calculator<2>::kIsotropicDirections;

template <>
const geometry::vectors2d variogram_calculator<2>::kAnisotropicDirections;

template <>
const geometry::vectors3d variogram_calculator<3>::kIsotropicDirections;

template <>
const geometry::vectors3d variogram_calculator<3>::kAnisotropicDirections;

}  // namespace polatory::kriging
