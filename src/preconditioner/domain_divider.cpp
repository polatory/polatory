// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/preconditioner/domain_divider.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>

#include <Eigen/Core>

#include <polatory/common/bsearch.hpp>
#include <polatory/common/eigen_utility.hpp>
#include <polatory/common/zip_sort.hpp>

namespace polatory {
namespace preconditioner {

size_t domain::size() const {
  return point_indices.size();
}

void domain::merge_poly_points(const std::vector<size_t>& poly_point_idcs) {
  std::vector<size_t> new_point_indices(poly_point_idcs.begin(), poly_point_idcs.end());
  std::vector<bool> new_inner_point(new_point_indices.size());

  common::zip_sort(
    point_indices.begin(), point_indices.end(),
    inner_point.begin(), inner_point.end(),
    [](const auto& a, const auto& b) { return a.first < b.first; }
  );

  for (size_t i = 0; i < poly_point_idcs.size(); i++) {
    auto it = common::bsearch_eq(point_indices.begin(), point_indices.end(), poly_point_idcs[i]);
    if (it == point_indices.end())
      continue;

    auto it_inner = inner_point.begin() + std::distance(point_indices.begin(), it);
    if (*it_inner) {
      new_inner_point[i] = true;
    }

    point_indices.erase(it);
    inner_point.erase(it_inner);
  }

  new_point_indices.insert(new_point_indices.end(), point_indices.begin(), point_indices.end());
  new_inner_point.insert(new_inner_point.end(), inner_point.begin(), inner_point.end());

  point_indices = new_point_indices;
  inner_point = new_inner_point;
}

domain_divider::domain_divider(const geometry::points3d& points,
                               const std::vector<size_t>& point_indices,
                               const std::vector<size_t>& poly_point_indices)
  : points_(points)
  , size_of_root_(point_indices.size())
  , poly_point_idcs_(poly_point_indices) {
  auto root = domain();

  root.point_indices = point_indices;

  root.inner_point = std::vector<bool>(point_indices.size(), true);

  root.bbox_ = domain_bbox(root);
  longest_side_length_of_root_ = root.bbox_.size().maxCoeff();

  domains_.push_back(root);

  divide_domains();
}

std::vector<size_t> domain_divider::choose_coarse_points(double ratio) const {
  std::vector<size_t> coarse_idcs(poly_point_idcs_.begin(), poly_point_idcs_.end());

  std::random_device rd;
  std::mt19937 gen(rd());

  for (const auto& d : domains_) {
    std::vector<size_t> shuffled(d.size() - poly_point_idcs_.size());
    std::iota(shuffled.begin(), shuffled.end(), poly_point_idcs_.size());
    std::shuffle(shuffled.begin(), shuffled.end(), gen);

    auto n_inner_pts = std::count(d.inner_point.begin(), d.inner_point.end(), true);
    auto n_coarse = std::max(size_t(1), static_cast<size_t>(round_half_to_even(ratio * n_inner_pts)));

    size_t count = 0;
    for (auto i : shuffled) {
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

const std::list<domain>& domain_divider::domains() const {
  return domains_;
}

void domain_divider::divide_domain(std::list<domain>::iterator it) {
  auto& d = *it;

  size_t split_axis;
  (void)d.bbox_.size().maxCoeff(&split_axis);

  common::zip_sort(
    d.point_indices.begin(), d.point_indices.end(),
    d.inner_point.begin(), d.inner_point.end(),
    [this, split_axis](const auto& a, const auto& b) {
      return points_(a.first, split_axis) < points_(b.first, split_axis);
    });

  auto longest_side_length = d.bbox_.size()(split_axis);
  auto q =
    longest_side_length_of_root_ / longest_side_length * std::sqrt(static_cast<double>(max_leaf_size) / static_cast<double>(size_of_root_)) *
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

  left.bbox_ = domain_bbox(left);

  domain right;
  right.point_indices = std::vector<size_t>(
    d.point_indices.begin() + left_partition,
    d.point_indices.end());

  right.inner_point = std::vector<bool>(
    d.inner_point.begin() + left_partition,
    d.inner_point.end());
  std::fill(right.inner_point.begin(), right.inner_point.begin() + (mid - left_partition), false);

  right.bbox_ = domain_bbox(right);

  domains_.push_back(left);
  domains_.push_back(right);
}

void domain_divider::divide_domains() {
  auto it = domains_.begin();

  while (it != domains_.end()) {
    auto& d = *it;
    if (d.size() <= max_leaf_size) {
      ++it;
      continue;
    }

    divide_domain(it);

    it = domains_.erase(it);
  }

  for (auto& d : domains_) {
    d.merge_poly_points(poly_point_idcs_);
  }
}

geometry::bbox3d domain_divider::domain_bbox(const domain& domain) const {
  auto domain_points = common::take_rows(points_, domain.point_indices);

  return geometry::bbox3d::from_points(domain_points);
}

double domain_divider::round_half_to_even(double d) {
  return std::ceil((d - 0.5) / 2.0) + std::floor((d + 0.5) / 2.0);
}

}  // namespace preconditioner
}  // namespace polatory
