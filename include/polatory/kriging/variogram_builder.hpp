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
  variogram_builder(double bin_interval, double bin_tolerance, index_t num_bins,
                    const Vector& direction)
      : bin_interval_(bin_interval),
        bin_tolerance_(bin_tolerance),
        num_bins_(num_bins),
        direction_(direction) {
    lag_.resize(num_bins, Vector::Zero());
    gamma_.resize(num_bins);
    num_pairs_.resize(num_bins);
  }

  void add_pair(const Point& point_i, const Point& point_j, double value_i, double value_j) {
    auto lag = point_j - point_i;
    auto dist = lag.norm();
    auto incr = value_j - value_i;
    auto gamma = 0.5 * (incr * incr);

    auto first = static_cast<index_t>(std::ceil((dist - bin_tolerance_) / bin_interval_));
    auto last = static_cast<index_t>(std::floor((dist + bin_tolerance_) / bin_interval_));
    first = std::max(first, index_t{0});
    last = std::min(last, num_bins_ - 1);

    for (index_t bin = first; bin <= last; bin++) {
      lag_.at(bin) += lag;
      gamma_.at(bin) += gamma;
      num_pairs_.at(bin)++;
    }
  }

  Variogram into_variogram() {
    auto l_it = lag_.begin();
    auto g_it = gamma_.begin();
    auto np_it = num_pairs_.begin();
    while (np_it != num_pairs_.end()) {
      auto np = *np_it;
      if (np == 0) {
        // Remove empty bin.
        l_it = lag_.erase(l_it);
        g_it = gamma_.erase(g_it);
        np_it = num_pairs_.erase(np_it);
      } else {
        *l_it /= static_cast<double>(np);
        *g_it /= static_cast<double>(np);

        ++l_it;
        ++g_it;
        ++np_it;
      }
    }

    return Variogram(std::move(lag_), std::move(gamma_), std::move(num_pairs_), direction_);
  }

  void merge(const variogram_builder& other) {
    POLATORY_ASSERT(other.bin_interval_ == bin_interval_);
    POLATORY_ASSERT(other.bin_tolerance_ == bin_tolerance_);
    POLATORY_ASSERT(other.num_bins_ == num_bins_);
    POLATORY_ASSERT(other.direction_ == direction_);

    for (index_t i = 0; i < num_bins_; i++) {
      lag_.at(i) += other.lag_.at(i);
      gamma_.at(i) += other.gamma_.at(i);
      num_pairs_.at(i) += other.num_pairs_.at(i);
    }
  }

 private:
  const double bin_interval_;
  const double bin_tolerance_;
  const index_t num_bins_;
  const Vector direction_;
  std::vector<Vector> lag_;
  std::vector<double> gamma_;
  std::vector<index_t> num_pairs_;
};

}  // namespace polatory::kriging