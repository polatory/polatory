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
  // Non-constexpr for the sake of Python bindings.
  static inline const double kAutomaticAngleTolerance = -1.0;

  static const Vectors kIsotropicDirections;
  static const Vectors kAnisotropicDirections;

  variogram_calculator(double lag_distance, index_t num_lags)
      : lag_distance_(lag_distance), num_lags_(num_lags), lag_tolerance_{0.5 * lag_distance} {}

  double angle_tolerance() const { return angle_tolerance_; }

  std::vector<Variogram> calculate(const Points& points, const common::valuesd& values) const {
    auto num_directions = directions_.rows();
    auto num_points = points.rows();
    auto squared_cos_angle_tolerance = std::pow(std::cos(angle_tolerance_), 2);

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

      common::valuesd squared_dots;

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
  Vectors directions_{kIsotropicDirections};
  double angle_tolerance_{kAutomaticAngleTolerance};
};

template <>
inline const geometry::vectors1d variogram_calculator<1>::kIsotropicDirections{
    geometry::vector1d::UnitX()};

template <>
inline const geometry::vectors1d variogram_calculator<1>::kAnisotropicDirections{
    geometry::vector1d::UnitX()};

template <>
inline const geometry::vectors2d variogram_calculator<2>::kIsotropicDirections{
    geometry::vector2d::UnitX()};

template <>
inline const geometry::vectors2d variogram_calculator<2>::kAnisotropicDirections{
    {1.0, 0.0}, {0.92387953, 0.38268343},  {0.70710678, 0.70710678},  {0.38268343, 0.92387953},
    {0.0, 1.0}, {-0.38268343, 0.92387953}, {-0.70710678, 0.70710678}, {-0.92387953, 0.38268343}};

template <>
inline const geometry::vectors3d variogram_calculator<3>::kIsotropicDirections{
    geometry::vector3d::UnitX()};

template <>
inline const geometry::vectors3d variogram_calculator<3>::kAnisotropicDirections{
    {0.0, 0.0, 1.0},
    {0.10607892, 0.32647735, 0.9392336},
    {-0.27771825, 0.20177411, 0.9392336},
    {-0.18759248, 0.57735026, 0.79465449},
    {-0.55543649, 0.40354821, 0.72707576},
    {-0.44935754, 0.73002559, 0.51491791},
    {-0.72360682, 0.52573109, 0.44721359},
    {0.21215785, 0.6529547, 0.72707576},
    {-0.065560378, 0.85472882, 0.51491791},
    {0.2763932, 0.85065079, 0.44721359},
    {-0.27771825, -0.20177411, 0.9392336},
    {-0.60706198, 0.0, 0.79465449},
    {-0.55543649, -0.40354821, 0.72707576},
    {-0.83315468, -0.20177411, 0.51491791},
    {-0.72360682, -0.52573109, 0.44721359},
    {-0.83315468, 0.20177411, 0.51491791},
    {0.10607892, -0.32647735, 0.9392336},
    {-0.18759248, -0.57735026, 0.79465449},
    {0.21215785, -0.6529547, 0.72707576},
    {-0.065560378, -0.85472882, 0.51491791},
    {0.2763932, -0.85065079, 0.44721359},
    {-0.44935754, -0.73002559, 0.51491791},
    {0.34327862, 0.0, 0.9392336},
    {0.49112347, -0.3568221, 0.79465449},
    {0.68655723, 0.0, 0.72707576},
    {0.79263616, -0.32647735, 0.51491791},
    {0.89442718, 0.0, 0.44721359},
    {0.55543649, -0.6529547, 0.51491791},
    {0.49112347, 0.3568221, 0.79465449},
    {0.55543649, 0.6529547, 0.51491791},
    {0.79263616, 0.32647735, 0.51491791},
    {0.10607892, 0.97943211, 0.17163931},
    {-0.30353099, 0.93417233, 0.18759248},
    {-0.66151541, 0.73002559, 0.17163931},
    {-0.89871508, 0.40354821, 0.17163931},
    {-0.98224694, 0.0, 0.18759248},
    {-0.89871508, -0.40354821, 0.17163931},
    {-0.66151541, -0.73002559, 0.17163931},
    {-0.30353099, -0.93417233, 0.18759248},
    {0.10607892, -0.97943211, 0.17163931},
    {0.48987609, -0.85472882, 0.17163931},
    {0.79465449, -0.57735026, 0.18759248},
    {0.96427548, -0.20177411, 0.17163931},
    {0.96427548, 0.20177411, 0.17163931},
    {0.79465449, 0.57735026, 0.18759248},
    {0.48987609, 0.85472882, 0.17163931}};

}  // namespace polatory::kriging
