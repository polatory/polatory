#pragma once

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <string>
#include <vector>

namespace polatory {
namespace kriging {

class empirical_variogram {
 public:
  empirical_variogram(const geometry::points3d& points, const common::valuesd& values,
                      double bin_width, index_t n_bins);

  explicit empirical_variogram(const std::string& filename);

  const std::vector<double>& bin_distance() const;

  const std::vector<double>& bin_gamma() const;

  const std::vector<index_t>& bin_num_pairs() const;

  void save(const std::string& filename) const;

 private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {  // NOLINT(runtime/references)
    ar& distance_;
    ar& gamma_;
    ar& num_pairs_;
  }

  std::vector<double> distance_;
  std::vector<double> gamma_;
  std::vector<index_t> num_pairs_;
};

}  // namespace kriging
}  // namespace polatory
