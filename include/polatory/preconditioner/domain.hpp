#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <iterator>
#include <polatory/common/zip_sort.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::preconditioner {

template <int Dim>
class DomainDivider;

template <int Dim>
class Domain {
 public:
  std::vector<Index> point_indices;
  std::vector<Index> grad_point_indices;
  std::vector<bool> inner_point;
  std::vector<bool> inner_grad_point;

  Index num_points() const { return static_cast<Index>(point_indices.size()); }

  Index num_grad_points() const { return static_cast<Index>(grad_point_indices.size()); }

 private:
  friend class DomainDivider<Dim>;

  void merge_poly_points(const std::vector<Index>& poly_point_idcs) {
    common::zip_sort(point_indices.begin(), point_indices.end(), inner_point.begin(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    auto n_poly_points = static_cast<Index>(poly_point_idcs.size());
    point_indices.insert(point_indices.begin(), poly_point_idcs.begin(), poly_point_idcs.end());
    inner_point.insert(inner_point.begin(), n_poly_points, false);

    for (Index i = 0; i < n_poly_points; i++) {
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
