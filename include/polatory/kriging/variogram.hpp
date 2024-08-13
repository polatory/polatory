#pragma once

#include <numeric>
#include <polatory/common/io.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/normal_score_transformation.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class Variogram {
  using Vector = geometry::Vector<Dim>;

 public:
  Variogram(std::vector<double>&& bin_distance, std::vector<double>&& bin_gamma,
            std::vector<Index>&& bin_num_pairs, const Vector& direction)
      : bin_distance_{std::move(bin_distance)},
        bin_gamma_{std::move(bin_gamma)},
        bin_num_pairs_{std::move(bin_num_pairs)},
        direction_{direction} {}

  void back_transform(const NormalScoreTransformation& t) {
    for (auto& gamma : bin_gamma_) {
      gamma = t.back_transform_gamma(gamma);
    }
  }

  const std::vector<double>& bin_distance() const { return bin_distance_; }

  const std::vector<double>& bin_gamma() const { return bin_gamma_; }

  const std::vector<Index>& bin_num_pairs() const { return bin_num_pairs_; }

  const Vector& direction() const { return direction_; }

  Index num_bins() const { return static_cast<Index>(bin_distance_.size()); }

  Index num_pairs() const { return std::reduce(bin_num_pairs_.begin(), bin_num_pairs_.end()); }

 private:
  POLATORY_FRIEND_READ_WRITE;

  // For deserialization.
  Variogram() = default;

  std::vector<double> bin_distance_;
  std::vector<double> bin_gamma_;
  std::vector<Index> bin_num_pairs_;
  Vector direction_;
};

}  // namespace polatory::kriging

namespace polatory::common {

template <int Dim>
struct Read<kriging::Variogram<Dim>> {
  void operator()(std::istream& is, kriging::Variogram<Dim>& t) const {
    read(is, t.bin_distance_);
    read(is, t.bin_gamma_);
    read(is, t.bin_num_pairs_);
    read(is, t.direction_);
  }
};

template <int Dim>
struct Write<kriging::Variogram<Dim>> {
  void operator()(std::ostream& os, const kriging::Variogram<Dim>& t) const {
    write(os, t.bin_distance_);
    write(os, t.bin_gamma_);
    write(os, t.bin_num_pairs_);
    write(os, t.direction_);
  }
};

}  // namespace polatory::common
