#pragma once

#include <polatory/common/io.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <string>
#include <vector>

namespace polatory::kriging {

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
  friend struct common::read<empirical_variogram>;
  friend struct common::write<empirical_variogram>;

  std::vector<double> distance_;
  std::vector<double> gamma_;
  std::vector<index_t> num_pairs_;
};

}  // namespace polatory::kriging

namespace polatory::common {

template <>
struct read<kriging::empirical_variogram> {
  void operator()(std::istream& is, kriging::empirical_variogram& t) const {
    index_t n_bins{};
    read<index_t>{}(is, n_bins);

    for (index_t i = 0; i < n_bins; ++i) {
      double distance{};
      double gamma{};
      index_t num_pairs{};
      read<double>{}(is, distance);
      read<double>{}(is, gamma);
      read<index_t>{}(is, num_pairs);
      t.distance_.push_back(distance);
      t.gamma_.push_back(gamma);
      t.num_pairs_.push_back(num_pairs);
    }
  }
};

template <>
struct write<kriging::empirical_variogram> {
  void operator()(std::ostream& os, const kriging::empirical_variogram& t) const {
    auto n_bins = static_cast<index_t>(t.distance_.size());
    write<index_t>{}(os, n_bins);

    for (index_t i = 0; i < n_bins; ++i) {
      write<double>{}(os, t.distance_.at(i));
      write<double>{}(os, t.gamma_.at(i));
      write<index_t>{}(os, t.num_pairs_.at(i));
    }
  }
};

}  // namespace polatory::common
