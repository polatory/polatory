// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <string>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include <polatory/common/types.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace kriging {

class empirical_variogram {
public:
  empirical_variogram(const geometry::points3d& points, const common::valuesd& values,
                      double bin_width, size_t n_bins);

  explicit empirical_variogram(const std::string& filename);

  const std::vector<double>& bin_distance() const;

  const std::vector<double>& bin_gamma() const;

  const std::vector<size_t>& bin_num_pairs() const;

  void save(const std::string& filename) const;

private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int) {  // NOLINT(runtime/references)
    ar & distance_;
    ar & gamma_;
    ar & num_pairs_;
  }

  std::vector<double> distance_;
  std::vector<double> gamma_;
  std::vector<size_t> num_pairs_;
};

}  // namespace kriging
}  // namespace polatory
