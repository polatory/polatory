// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <list>
#include <vector>

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory {
namespace preconditioner {

class domain_divider;

class domain {
public:
  std::vector<size_t> point_indices;
  std::vector<bool> inner_point;

  size_t size() const;

private:
  friend class domain_divider;

  void merge_poly_points(const std::vector<size_t>& poly_point_idcs);

  geometry::bbox3d bbox_;
};

class domain_divider {
  static constexpr double overlap_quota = 0.75;
  static constexpr size_t max_leaf_size = 256;

public:
  domain_divider(const geometry::points3d& points,
                 const std::vector<size_t>& point_indices,
                 const std::vector<size_t>& poly_point_indices);

  std::vector<size_t> choose_coarse_points(double ratio) const;

  const std::list<domain>& domains() const;

private:
  void divide_domain(std::list<domain>::iterator it);

  void divide_domains();

  geometry::bbox3d domain_bbox(const domain& domain) const;

  static double round_half_to_even(double d);

  const geometry::points3d& points_;

  size_t size_of_root_;
  double longest_side_length_of_root_;
  std::vector<size_t> poly_point_idcs_;
  std::list<domain> domains_;
};

} // namespace preconditioner
} // namespace polatory
