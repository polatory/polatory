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
struct Domain {
  std::vector<Index> point_indices;
  std::vector<Index> grad_point_indices;
  std::vector<bool> inner_point;
  std::vector<bool> inner_grad_point;

  Index num_points() const { return static_cast<Index>(point_indices.size()); }

  Index num_grad_points() const { return static_cast<Index>(grad_point_indices.size()); }
};

}  // namespace polatory::preconditioner
