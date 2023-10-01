#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <iterator>
#include <polatory/common/zip_sort.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/types.hpp>
#include <vector>

namespace polatory::preconditioner {

template <int Dim>
class domain_divider;

template <int Dim>
class domain {
  using Bbox = geometry::bboxNd<Dim>;

 public:
  std::vector<index_t> point_indices;
  std::vector<index_t> grad_point_indices;
  std::vector<bool> inner_point;
  std::vector<bool> inner_grad_point;

  index_t size() const { return static_cast<index_t>(point_indices.size()); }

  index_t grad_size() const { return static_cast<index_t>(grad_point_indices.size()); }

  index_t mixed_size() const { return size() + grad_size(); }

 private:
  friend class domain_divider<Dim>;

  void merge_poly_points(const std::vector<index_t>& poly_point_idcs) {
    common::zip_sort(point_indices.begin(), point_indices.end(), inner_point.begin(),
                     inner_point.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    auto n_poly_points = static_cast<index_t>(poly_point_idcs.size());
    std::vector<index_t> new_point_indices(poly_point_idcs);
    std::vector<bool> new_inner_point(n_poly_points);

    for (index_t i = 0; i < n_poly_points; i++) {
      auto idx = poly_point_idcs.at(i);
      auto it = std::lower_bound(point_indices.begin(), point_indices.end(), idx);
      if (it == point_indices.end() || *it != idx) {
        continue;
      }

      auto it_inner = inner_point.begin() + std::distance(point_indices.begin(), it);
      new_inner_point.at(i) = *it_inner;

      point_indices.erase(it);
      inner_point.erase(it_inner);
    }

    new_point_indices.insert(new_point_indices.end(), point_indices.begin(), point_indices.end());
    new_inner_point.insert(new_inner_point.end(), inner_point.begin(), inner_point.end());

    point_indices = new_point_indices;
    inner_point = new_inner_point;
  }

  Bbox bbox_;
};

}  // namespace polatory::preconditioner
