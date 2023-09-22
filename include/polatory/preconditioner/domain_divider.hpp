#pragma once

#include <list>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/types.hpp>
#include <utility>
#include <vector>

namespace polatory::preconditioner {

class domain_divider {
  static constexpr double overlap_quota = 0.75;
  static constexpr index_t max_leaf_size = 256;

 public:
  domain_divider(const geometry::points3d& points, const geometry::points3d& grad_points,
                 const std::vector<index_t>& point_indices,
                 const std::vector<index_t>& grad_point_indices,
                 const std::vector<index_t>& poly_point_indices);

  std::pair<std::vector<index_t>, std::vector<index_t>> choose_coarse_points(double ratio) const;

  const std::list<domain>& domains() const;

  std::list<domain>&& into_domains();

 private:
  void divide_domain(std::list<domain>::iterator it);

  void divide_domains();

  geometry::bbox3d domain_bbox(const domain& domain) const;

  static double round_half_to_even(double d);

  const geometry::points3d& points_;
  const geometry::points3d& grad_points_;

  index_t mixed_size_of_root_;
  double longest_side_length_of_root_;
  std::vector<index_t> poly_point_idcs_;
  std::list<domain> domains_;
};

}  // namespace polatory::preconditioner
