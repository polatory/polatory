#pragma once

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class variogram {
  using Vector = geometry::vectorNd<Dim>;

 public:
  variogram(std::vector<Vector>&& bin_lag, std::vector<double>&& bin_gamma,
            std::vector<index_t>&& bin_num_pairs, const Vector& direction)
      : bin_lag_{std::move(bin_lag)},
        bin_gamma_{std::move(bin_gamma)},
        bin_num_pairs_{std::move(bin_num_pairs)},
        direction_{direction} {}

  const std::vector<Vector>& bin_lag() const { return bin_lag_; }

  const std::vector<double>& bin_gamma() const { return bin_gamma_; }

  const std::vector<index_t>& bin_num_pairs() const { return bin_num_pairs_; }

  const Vector& direction() const { return direction_; }

  index_t num_bins() const { return static_cast<index_t>(bin_lag_.size()); }

 private:
  std::vector<Vector> bin_lag_;
  std::vector<double> bin_gamma_;
  std::vector<index_t> bin_num_pairs_;
  Vector direction_;
};

}  // namespace polatory::kriging
