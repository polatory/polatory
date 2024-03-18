#include <cmath>
#include <fstream>
#include <polatory/common/macros.hpp>
#include <polatory/kriging/empirical_variogram.hpp>

namespace polatory::kriging {

empirical_variogram::empirical_variogram(const geometry::points3d& points,
                                         const common::valuesd& values, double bin_width,
                                         index_t n_bins) {
  POLATORY_ASSERT(values.rows() == points.rows());

  auto n_points = points.rows();
  if (n_points == 0) {
    return;
  }

  distance_.resize(n_bins);
  gamma_.resize(n_bins);
  num_pairs_.resize(n_bins);

#pragma omp parallel
  {
    std::vector<double> distance_local(n_bins);
    std::vector<double> gamma_local(n_bins);
    std::vector<index_t> num_pairs_local(n_bins);

#pragma omp for schedule(dynamic)
    for (index_t i = 0; i < n_points - 1; i++) {
      for (index_t j = i + 1; j < n_points; j++) {
        auto dist = (points.row(i) - points.row(j)).norm();
        // gstat's convention (to include more pairs in the first bin?):
        //   https://github.com/edzer/gstat/blob/32003307b11d6354340b653ab67c2d85d7304824/src/sem.c#L734-L738
        auto frac = dist / bin_width;
        auto bin = dist > 0.0 && frac == std::floor(frac)
                       ? static_cast<index_t>(std::floor(frac)) - 1
                       : static_cast<index_t>(std::floor(frac));
        if (bin >= n_bins) {
          continue;
        }

        distance_local.at(bin) += dist;
        gamma_local.at(bin) += std::pow(values(i) - values(j), 2.0) / 2.0;
        num_pairs_local.at(bin)++;
      }
    }

#pragma omp critical
    {
      for (index_t i = 0; i < n_bins; i++) {
        distance_.at(i) += distance_local.at(i);
        gamma_.at(i) += gamma_local.at(i);
        num_pairs_.at(i) += num_pairs_local.at(i);
      }
    }
  }

  for (index_t i = 0; i < n_bins; i++) {
    if (num_pairs_.at(i) == 0) {
      continue;
    }

    distance_.at(i) /= static_cast<double>(num_pairs_.at(i));
    gamma_.at(i) /= static_cast<double>(num_pairs_.at(i));
  }

  // Remove empty bins.
  auto d_it = distance_.begin();
  auto g_it = gamma_.begin();
  auto np_it = num_pairs_.begin();
  while (np_it != num_pairs_.end()) {
    if (*np_it == 0) {
      d_it = distance_.erase(d_it);
      g_it = gamma_.erase(g_it);
      np_it = num_pairs_.erase(np_it);
    } else {
      ++d_it;
      ++g_it;
      ++np_it;
    }
  }
}

empirical_variogram::empirical_variogram(const std::string& filename) {
  common::load(filename, *this);
}

const std::vector<double>& empirical_variogram::bin_distance() const { return distance_; }

const std::vector<double>& empirical_variogram::bin_gamma() const { return gamma_; }

const std::vector<index_t>& empirical_variogram::bin_num_pairs() const { return num_pairs_; }

void empirical_variogram::save(const std::string& filename) const { common::save(filename, *this); }

}  // namespace polatory::kriging
