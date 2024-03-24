#pragma once

#include <cmath>
#include <polatory/common/macros.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/variogram.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class variogram_builder {
  static constexpr int kDim = Dim;
  using Point = geometry::pointNd<kDim>;
  using Variogram = variogram<kDim>;
  using Vector = geometry::vectorNd<Dim>;

 public:
  variogram_builder(double bin_width, index_t num_bins, const Vector& direction)
      : bin_width_(bin_width), num_bins_(num_bins), direction_(direction) {
    distance_.resize(num_bins);
    gamma_.resize(num_bins);
    num_pairs_.resize(num_bins);
  }

  void add_pair(const Point& point_i, const Point& point_j, double value_i, double value_j) {
    auto dist = (point_j - point_i).norm();

    auto bin = static_cast<index_t>(std::floor(dist / bin_width_));
    if (bin >= num_bins_) {
      return;
    }

    auto incr = value_j - value_i;

    distance_.at(bin) += dist;
    gamma_.at(bin) += (incr * incr) / 2.0;
    num_pairs_.at(bin)++;
  }

  Variogram into_variogram() {
    auto d_it = distance_.begin();
    auto g_it = gamma_.begin();
    auto np_it = num_pairs_.begin();
    while (np_it != num_pairs_.end()) {
      auto np = *np_it;
      if (np == 0) {
        // Remove empty bin.
        d_it = distance_.erase(d_it);
        g_it = gamma_.erase(g_it);
        np_it = num_pairs_.erase(np_it);
      } else {
        *d_it /= static_cast<double>(np);
        *g_it /= static_cast<double>(np);

        ++d_it;
        ++g_it;
        ++np_it;
      }
    }

    return Variogram(std::move(distance_), std::move(gamma_), std::move(num_pairs_), direction_);
  }

  void merge(const variogram_builder& other) {
    POLATORY_ASSERT(other.bin_width_ == bin_width_);
    POLATORY_ASSERT(other.num_bins_ == num_bins_);
    POLATORY_ASSERT(other.direction_ == direction_);

    for (index_t i = 0; i < num_bins_; i++) {
      distance_.at(i) += other.distance_.at(i);
      gamma_.at(i) += other.gamma_.at(i);
      num_pairs_.at(i) += other.num_pairs_.at(i);
    }
  }

 private:
  const double bin_width_;
  const index_t num_bins_;
  const Vector direction_;
  std::vector<double> distance_;
  std::vector<double> gamma_;
  std::vector<index_t> num_pairs_;
};

}  // namespace polatory::kriging