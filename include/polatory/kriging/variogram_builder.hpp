#pragma once

#include <algorithm>
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
  variogram_builder(double lag_distance, double lag_tolerance, index_t num_lags,
                    const Vector& direction)
      : lag_distance_(lag_distance),
        inv_lag_distance_(1.0 / lag_distance),
        lag_tolerance_(lag_tolerance),
        num_lags_(num_lags),
        direction_(direction) {
    bin_distance_.resize(num_lags);
    bin_gamma_.resize(num_lags);
    bin_num_pairs_.resize(num_lags);
  }

  void add_pair(const Point& point_i, const Point& point_j, double value_i, double value_j) {
    auto lag = point_j - point_i;
    auto dist = lag.norm();
    auto incr = value_j - value_i;
    auto gamma = 0.5 * (incr * incr);

    auto first = static_cast<index_t>(std::ceil(inv_lag_distance_ * (dist - lag_tolerance_)));
    auto last = static_cast<index_t>(std::floor(inv_lag_distance_ * (dist + lag_tolerance_)));
    first = std::max(first, index_t{0});
    last = std::min(last, num_lags_ - 1);

    for (index_t bin = first; bin <= last; bin++) {
      bin_distance_.at(bin) += dist;
      bin_gamma_.at(bin) += gamma;
      bin_num_pairs_.at(bin)++;
    }
  }

  Variogram into_variogram() {
    auto d_it = bin_distance_.begin();
    auto g_it = bin_gamma_.begin();
    auto np_it = bin_num_pairs_.begin();
    while (np_it != bin_num_pairs_.end()) {
      auto np = *np_it;
      if (np == 0) {
        // Remove empty bin.
        d_it = bin_distance_.erase(d_it);
        g_it = bin_gamma_.erase(g_it);
        np_it = bin_num_pairs_.erase(np_it);
      } else {
        *d_it /= static_cast<double>(np);
        *g_it /= static_cast<double>(np);

        ++d_it;
        ++g_it;
        ++np_it;
      }
    }

    return Variogram(std::move(bin_distance_), std::move(bin_gamma_), std::move(bin_num_pairs_),
                     direction_);
  }

  void merge(const variogram_builder& other) {
    POLATORY_ASSERT(other.lag_distance_ == lag_distance_);
    POLATORY_ASSERT(other.lag_tolerance_ == lag_tolerance_);
    POLATORY_ASSERT(other.num_lags_ == num_lags_);
    POLATORY_ASSERT(other.direction_ == direction_);

    for (index_t i = 0; i < num_lags_; i++) {
      bin_distance_.at(i) += other.bin_distance_.at(i);
      bin_gamma_.at(i) += other.bin_gamma_.at(i);
      bin_num_pairs_.at(i) += other.bin_num_pairs_.at(i);
    }
  }

 private:
  const double lag_distance_;
  const double inv_lag_distance_;
  const double lag_tolerance_;
  const index_t num_lags_;
  const Vector direction_;
  std::vector<double> bin_distance_;
  std::vector<double> bin_gamma_;
  std::vector<index_t> bin_num_pairs_;
};

}  // namespace polatory::kriging