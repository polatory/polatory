#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/variogram.hpp>
#include <polatory/kriging/variogram_builder.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::kriging {

inline const geometry::vectors3d kDirections{{0.0, 0.0, 1.0},
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

template <int Dim>
class variogram_calculator {
  static constexpr int kDim = Dim;
  using Points = geometry::pointsNd<kDim>;
  using Variogram = variogram<kDim>;
  using VariogramBuilder = variogram_builder<kDim>;
  using Vector = geometry::vectorNd<kDim>;
  using Vectors = geometry::vectorsNd<kDim>;

 public:
  variogram_calculator(const Points& points, const common::valuesd& values, double bin_interval,
                       double bin_tolerance, index_t num_bins)
      : variogram_calculator(points, values, bin_interval, bin_tolerance, num_bins,
                             std::numbers::pi / 2.0, Vector::UnitX()) {}

  variogram_calculator(const Points& points, const common::valuesd& values, double bin_interval,
                       double bin_tolerance, index_t num_bins, double angle_tolerance,
                       Vectors directions) {
    auto cos_angle_tolerance = std::cos(angle_tolerance);
    auto num_directions = directions.rows();

    std::vector<VariogramBuilder> builders;
    for (index_t k = 0; k < num_directions; k++) {
      builders.emplace_back(bin_interval, bin_tolerance, num_bins, directions.row(k));
    }

#pragma omp parallel
    {
      std::vector<VariogramBuilder> local_builders;
      for (index_t k = 0; k < num_directions; k++) {
        local_builders.emplace_back(bin_interval, bin_tolerance, num_bins, directions.row(k));
      }

#pragma omp for schedule(dynamic)
      for (index_t i = 0; i < points.rows() - 1; i++) {
        for (index_t j = i + 1; j < points.rows(); j++) {
          auto point_i = points.row(i);
          auto point_j = points.row(j);
          auto value_i = values(i);
          auto value_j = values(j);

          auto dir = (point_j - point_i).normalized();
          for (index_t k = 0; k < num_directions; k++) {
            if (std::abs(dir.dot(directions.row(k))) >= cos_angle_tolerance) {
              local_builders.at(k).add_pair(point_i, point_j, value_i, value_j);
            }
          }
        }
      }

#pragma omp critical
      for (index_t k = 0; k < num_directions; k++) {
        builders.at(k).merge(local_builders.at(k));
      }
    }

    for (auto& builder : builders) {
      variograms_.emplace_back(builder.into_variogram());
    }
  }

  const std::vector<Variogram>& variograms() { return variograms_; }

 private:
  std::vector<Variogram> variograms_;
};

}  // namespace polatory::kriging