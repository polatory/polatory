#pragma once

#include <cmath>
#include <numbers>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/variogram.hpp>
#include <polatory/kriging/variogram_builder.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class variogram_calculator {
  static constexpr int kDim = Dim;
  using Points = geometry::pointsNd<kDim>;
  using Variogram = variogram<kDim>;
  using VariogramBuilder = variogram_builder<kDim>;
  using Vector = geometry::vectorNd<kDim>;
  using Vectors = geometry::vectorsNd<kDim>;

 public:
  // Not constexpr because of Python bindings.
  static inline const double kAutomaticAngleTolerance = -1.0;

  variogram_calculator(double lag_distance, index_t num_lags)
      : lag_distance_(lag_distance), num_lags_(num_lags), lag_tolerance_{0.5 * lag_distance} {}

  double angle_tolerance() const { return angle_tolerance_; }

  std::vector<Variogram> calculate(const Points& points, const common::valuesd& values) const {
    auto num_directions = directions_.rows();
    auto num_points = points.rows();
    auto cos_angle_tolerance = std::cos(angle_tolerance_);

    std::vector<VariogramBuilder> builders;
    for (index_t k = 0; k < num_directions; k++) {
      builders.emplace_back(lag_distance_, lag_tolerance_, num_lags_, directions_.row(k));
    }

#pragma omp parallel
    {
      std::vector<VariogramBuilder> local_builders;
      for (index_t k = 0; k < num_directions; k++) {
        local_builders.emplace_back(lag_distance_, lag_tolerance_, num_lags_, directions_.row(k));
      }

      common::valuesd dots;
      common::valuesd abs_dots;

#pragma omp for schedule(dynamic)
      for (index_t i = 0; i < num_points - 1; i++) {
        auto point_i = points.row(i);
        auto value_i = values(i);

        for (index_t j = i + 1; j < num_points; j++) {
          auto point_j = points.row(j);
          auto value_j = values(j);

          // Do not normalize the direction to avoid division for performance.
          Vector dir = point_j - point_i;
          dots = dir * directions_.transpose();
          abs_dots = dots.cwiseAbs();

          if (angle_tolerance_ == kAutomaticAngleTolerance) {
            index_t k{};
            abs_dots.maxCoeff(&k);
            if (dots(k) > 0.0) {
              local_builders.at(k).add_pair(point_i, point_j, value_i, value_j);
            } else {
              local_builders.at(k).add_pair(point_j, point_i, value_j, value_i);
            }
          } else {
            auto threshold = dir.norm() * cos_angle_tolerance;
            for (index_t k = 0; k < num_directions; k++) {
              if (abs_dots(k) >= threshold) {
                if (dots(k) > 0.0) {
                  local_builders.at(k).add_pair(point_i, point_j, value_i, value_j);
                } else {
                  local_builders.at(k).add_pair(point_j, point_i, value_j, value_i);
                }
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
      variograms.emplace_back(builder.into_variogram());
    }

    return variograms;
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

    directions_ = directions;
  }

  void set_lag_tolerance(double lag_tolerance) {
    if (!(lag_tolerance > 0.0)) {
      throw std::invalid_argument("lag_tolerance must be positive");
    }

    lag_tolerance_ = lag_tolerance;
  }

 private:
  double lag_distance_;
  index_t num_lags_;
  double lag_tolerance_;
  Vectors directions_{Vector::UnitX()};
  double angle_tolerance_{kAutomaticAngleTolerance};
};

}  // namespace polatory::kriging