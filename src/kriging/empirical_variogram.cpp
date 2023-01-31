#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <cmath>
#include <fstream>
#include <polatory/common/macros.hpp>
#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/numeric/sum_accumulator.hpp>

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
  std::vector<numeric::kahan_sum_accumulator<double>> dist_sum(n_bins);
  std::vector<numeric::kahan_sum_accumulator<double>> gamma_sum(n_bins);

  for (index_t i = 0; i < n_points - 1; i++) {
    for (index_t j = i + 1; j < n_points; j++) {
      auto dist = (points.row(i) - points.row(j)).norm();
      // gstat's convention (to include more pairs in the first bin?):
      //   https://github.com/edzer/gstat/blob/32003307b11d6354340b653ab67c2d85d7304824/src/sem.c#L734-L738
      auto frac = dist / bin_width;
      auto bin = dist > 0.0 && frac == std::floor(frac) ? static_cast<index_t>(std::floor(frac)) - 1
                                                        : static_cast<index_t>(std::floor(frac));
      if (bin >= n_bins) {
        continue;
      }

      dist_sum.at(bin) += dist;
      gamma_sum.at(bin) += std::pow(values(i) - values(j), 2.0) / 2.0;
      num_pairs_.at(bin)++;
    }
  }

  for (index_t i = 0; i < n_bins; i++) {
    if (num_pairs_.at(i) == 0) {
      continue;
    }

    distance_.at(i) = dist_sum.at(i).get() / static_cast<double>(num_pairs_.at(i));
    gamma_.at(i) = gamma_sum.at(i).get() / static_cast<double>(num_pairs_.at(i));
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
  std::ifstream ifs(filename);
  boost::archive::text_iarchive ia(ifs);
  ia >> *this;
}

const std::vector<double>& empirical_variogram::bin_distance() const { return distance_; }

const std::vector<double>& empirical_variogram::bin_gamma() const { return gamma_; }

const std::vector<index_t>& empirical_variogram::bin_num_pairs() const { return num_pairs_; }

void empirical_variogram::save(const std::string& filename) const {
  std::ofstream ofs(filename);
  boost::archive::text_oarchive oa(ofs);
  oa << *this;
}

}  // namespace polatory::kriging
