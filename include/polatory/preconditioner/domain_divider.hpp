// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <list>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "polatory/common/vector_view.hpp"
#include "polatory/common/zip_sort.hpp"
#include "polatory/geometry/bbox3d.hpp"

namespace polatory {
namespace preconditioner {

struct domain {
  std::vector<size_t> point_indices;
  std::vector<bool> inner_point;
  geometry::bbox3d bbox;

  size_t size() const {
    return point_indices.size();
  }
};

class domain_divider {
  const double overlap_quota = 0.75;
  const size_t max_leaf_size = 256;

  const std::vector<Eigen::Vector3d>& points;

  size_t size_of_root;
  double longest_side_length_of_root;
  std::list<domain> domains_;

  void divide_domains() {
    auto it = domains_.begin();

    while (it != domains_.end()) {
      auto& d = *it;
      if (d.size() <= max_leaf_size) {
        ++it;
        continue;
      }

      size_t split_axis;
      d.bbox.size().maxCoeff(&split_axis);

      common::zip_sort(
        d.point_indices.begin(), d.point_indices.end(),
        d.inner_point.begin(), d.inner_point.end(),
        [this, &d, split_axis](const auto& a, const auto& b) {
          return points[a.first](split_axis) < points[b.first](split_axis);
        });

      auto longest_side_length = d.bbox.size()(split_axis);
      auto q =
        longest_side_length_of_root / longest_side_length * std::sqrt(double(max_leaf_size) / double(size_of_root)) *
        overlap_quota;
      q = std::min(0.5, q);

      auto n_pts = d.size();
      auto n_overlap_pts = static_cast<size_t>(round_half_to_even(q * n_pts));
      auto n_subdomain_pts = static_cast<size_t>(std::ceil((n_pts + n_overlap_pts) / 2.0));
      auto left_partition = n_pts - n_subdomain_pts;
      auto right_partition = n_subdomain_pts;
      auto mid = static_cast<size_t>(round_half_to_even((left_partition + right_partition) / 2.0));

      domain left;
      left.point_indices = std::vector<size_t>(
        d.point_indices.begin(),
        d.point_indices.begin() + right_partition);

      left.inner_point = std::vector<bool>(
        d.inner_point.begin(),
        d.inner_point.begin() + right_partition);
      std::fill(left.inner_point.begin() + mid, left.inner_point.end(), false);

      left.bbox = domain_bbox(left);

      domain right;
      right.point_indices = std::vector<size_t>(
        d.point_indices.begin() + left_partition,
        d.point_indices.end());

      right.inner_point = std::vector<bool>(
        d.inner_point.begin() + left_partition,
        d.inner_point.end());
      std::fill(right.inner_point.begin(), right.inner_point.begin() + (mid - left_partition), false);

      right.bbox = domain_bbox(right);

      domains_.push_back(left);
      domains_.push_back(right);

      it = domains_.erase(it);
    }
  }

  geometry::bbox3d domain_bbox(const domain& domain) const {
    auto domain_points = common::make_view(points, domain.point_indices);

    return geometry::bbox3d::from_points(domain_points);
  }

  static double round_half_to_even(double d) {
    return std::ceil((d - 0.5) / 2.0) + std::floor((d + 0.5) / 2.0);
  }

public:
  explicit domain_divider(const std::vector<Eigen::Vector3d>& points)
    : points(points)
    , size_of_root(points.size()) {
    auto root = domain();

    root.point_indices.resize(points.size());
    std::iota(root.point_indices.begin(), root.point_indices.end(), 0);

    root.inner_point = std::vector<bool>(points.size(), true);

    root.bbox = domain_bbox(root);
    longest_side_length_of_root = root.bbox.size().maxCoeff();

    domains_.push_back(root);

    divide_domains();
  }

  domain_divider(const std::vector<Eigen::Vector3d>& points, const std::vector<size_t>& point_indices)
    : points(points)
    , size_of_root(point_indices.size()) {
    auto root = domain();

    root.point_indices = point_indices;

    root.inner_point = std::vector<bool>(point_indices.size(), true);

    root.bbox = domain_bbox(root);
    longest_side_length_of_root = root.bbox.size().maxCoeff();

    domains_.push_back(root);

    divide_domains();
  }

  std::vector<size_t> choose_coarse_points(double ratio) const {
    std::vector<size_t> coarse_idcs;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto& d : domains_) {
      std::vector<size_t> permutation(d.size());
      std::iota(permutation.begin(), permutation.end(), 0);
      std::shuffle(permutation.begin(), permutation.end(), gen);

      auto n_inner_pts = std::count(d.inner_point.begin(), d.inner_point.end(), true);
      auto n_coarse = std::max(size_t(1), static_cast<size_t>(round_half_to_even(ratio * n_inner_pts)));

      size_t count = 0;
      for (auto i : permutation) {
        if (count == n_coarse)
          break;

        if (d.inner_point[i]) {
          coarse_idcs.push_back(d.point_indices[i]);
          count++;
        }
      }
    }

    return coarse_idcs;
  }

  const std::list<domain>& domains() const {
    return domains_;
  }
};

} // namespace preconditioner
} // namespace polatory