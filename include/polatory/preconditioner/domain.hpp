#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <iterator>
#include <polatory/common/zip_sort.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::preconditioner {

template <int Dim>
class domain_divider;

template <int Dim>
class domain {
 public:
  std::vector<index_t> point_indices;
  std::vector<index_t> grad_point_indices;
  std::vector<bool> inner_point;
  std::vector<bool> inner_grad_point;

  index_t num_points() const { return static_cast<index_t>(point_indices.size()); }

  index_t num_grad_points() const { return static_cast<index_t>(grad_point_indices.size()); }

 private:
  friend class domain_divider<Dim>;

  void merge_poly_points(const std::vector<index_t>& poly_point_idcs) {
    common::zip_sort(point_indices.begin(), point_indices.end(), inner_point.begin(),
                     inner_point.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    auto n_poly_points = static_cast<index_t>(poly_point_idcs.size());
    point_indices.insert(point_indices.begin(), poly_point_idcs.begin(), poly_point_idcs.end());
    inner_point.insert(inner_point.begin(), n_poly_points, false);

    for (index_t i = 0; i < n_poly_points; i++) {
      auto idx = point_indices.at(i);
      auto it = std::lower_bound(point_indices.begin() + n_poly_points, point_indices.end(), idx);
      if (it == point_indices.end() || *it != idx) {
        continue;
      }

      auto it_inner = inner_point.begin() + std::distance(point_indices.begin(), it);
      inner_point.at(i) = *it_inner;

      point_indices.erase(it);
      inner_point.erase(it_inner);
    }
  }
};

}  // namespace polatory::preconditioner
