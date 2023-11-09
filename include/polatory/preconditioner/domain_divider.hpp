#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <iterator>
#include <list>
#include <numeric>
#include <polatory/common/zip_sort.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/preconditioner/domain.hpp>
#include <polatory/types.hpp>
#include <random>
#include <utility>
#include <vector>

namespace polatory::preconditioner {

template <int Dim>
class domain_divider {
  static constexpr double kOverlapQuota = 0.5;
  static constexpr index_t kMaxLeafSize = 1024;

  using Bbox = geometry::bboxNd<Dim>;
  using Points = geometry::pointsNd<Dim>;
  using Domain = domain<Dim>;

 public:
  template <class DerivedPoints, class DerivedGradPoints>
  domain_divider(const Eigen::MatrixBase<DerivedPoints>& points,
                 const Eigen::MatrixBase<DerivedGradPoints>& grad_points,
                 const std::vector<index_t>& point_indices,
                 const std::vector<index_t>& grad_point_indices,
                 const std::vector<index_t>& poly_point_indices)
      : points_(points), grad_points_(grad_points), poly_point_idcs_(poly_point_indices) {
    Domain root;

    root.point_indices = point_indices;
    root.grad_point_indices = grad_point_indices;

    root.inner_point = std::vector<bool>(point_indices.size(), true);
    root.inner_grad_point = std::vector<bool>(grad_point_indices.size(), true);

    domains_.push_back(std::move(root));

    divide_domains();
  }

  std::pair<std::vector<index_t>, std::vector<index_t>> choose_coarse_points(double ratio) const {
    std::vector<index_t> idcs(poly_point_idcs_);
    std::vector<index_t> grad_idcs;

    std::random_device rd;
    std::mt19937 gen(rd());

    auto n_poly_points = static_cast<index_t>(poly_point_idcs_.size());
    for (const auto& d : domains_) {
      std::vector<mixed_point> mixed_points;
      for (index_t i = n_poly_points; i < d.size(); i++) {
        if (d.inner_point.at(i)) {
          mixed_points.emplace_back(d.point_indices.at(i), true, false);
        }
      }
      for (index_t i = 0; i < d.grad_size(); i++) {
        if (d.inner_grad_point.at(i)) {
          mixed_points.emplace_back(d.grad_point_indices.at(i), true, true);
        }
      }

      std::shuffle(mixed_points.begin(), mixed_points.end(), gen);

      auto n_inner_pts = std::count(d.inner_point.begin(), d.inner_point.end(), true) +
                         std::count(d.inner_grad_point.begin(), d.inner_grad_point.end(), true);
      auto n_coarse =
          static_cast<index_t>(round_half_to_even(ratio * static_cast<double>(n_inner_pts)));

      auto count = index_t{0};
      for (const auto& p : mixed_points) {
        if (count == n_coarse) {
          break;
        }

        (p.grad ? grad_idcs : idcs).push_back(p.index);
        count++;
      }
    }

    return {std::move(idcs), std::move(grad_idcs)};
  }

  const std::list<Domain>& domains() const { return domains_; }

  std::list<Domain>&& into_domains() { return std::move(domains_); }

 private:
  struct mixed_point {
    index_t index{};
    bool inner{};
    bool grad{};
  };

  void divide_domain(std::list<Domain>::iterator it) {
    auto& d = *it;

    std::vector<mixed_point> mixed_points;
    for (index_t i = 0; i < d.size(); i++) {
      mixed_points.emplace_back(d.point_indices.at(i), d.inner_point.at(i), false);
    }
    for (index_t i = 0; i < d.grad_size(); i++) {
      mixed_points.emplace_back(d.grad_point_indices.at(i), d.inner_grad_point.at(i), true);
    }

    auto bbox = domain_bbox(d);
    auto split_axis = index_t{0};
    bbox.width().maxCoeff(&split_axis);

    // TODO(mizuno): Sort all points along each axis and cache the result as a permutation.
    std::sort(mixed_points.begin(), mixed_points.end(),
              [this, split_axis](const auto& a, const auto& b) {
                return (a.grad ? grad_points_ : points_)(a.index, split_axis) <
                       (b.grad ? grad_points_ : points_)(b.index, split_axis);
              });

    auto q = kOverlapQuota * static_cast<double>(kMaxLeafSize) /
             static_cast<double>(mixed_points.size());

    auto n_pts = d.mixed_size();
    auto n_overlap_pts = static_cast<index_t>(round_half_to_even(q * static_cast<double>(n_pts)));
    auto n_subdomain_pts =
        static_cast<index_t>(std::ceil(static_cast<double>(n_pts + n_overlap_pts) / 2.0));
    auto left_partition = n_pts - n_subdomain_pts;
    auto right_partition = n_subdomain_pts;
    auto mid = static_cast<index_t>(
        round_half_to_even(static_cast<double>(left_partition + right_partition) / 2.0));

    Domain left;
    Domain right;

    for (index_t i = 0; i < right_partition; i++) {
      const auto& p = mixed_points.at(i);
      auto inner = p.inner && i < mid;

      if (p.grad) {
        left.grad_point_indices.push_back(p.index);
        left.inner_grad_point.push_back(inner);
      } else {
        left.point_indices.push_back(p.index);
        left.inner_point.push_back(inner);
      }
    }

    for (index_t i = left_partition; i < n_pts; i++) {
      const auto& p = mixed_points.at(i);
      auto inner = p.inner && i >= mid;

      if (p.grad) {
        right.grad_point_indices.push_back(p.index);
        right.inner_grad_point.push_back(inner);
      } else {
        right.point_indices.push_back(p.index);
        right.inner_point.push_back(inner);
      }
    }

    domains_.push_back(std::move(left));
    domains_.push_back(std::move(right));
  }

  void divide_domains() {
    auto it = domains_.begin();

    while (it != domains_.end()) {
      auto& d = *it;
      if (d.mixed_size() <= kMaxLeafSize) {
        ++it;
        continue;
      }

      divide_domain(it);

      it = domains_.erase(it);
    }

    remove_domains_with_no_inner_points();

    for (auto& d : domains_) {
      d.merge_poly_points(poly_point_idcs_);
    }
  }

  void remove_domains_with_no_inner_points() {
    auto it = domains_.begin();

    while (it != domains_.end()) {
      auto& d = *it;
      auto n_inner_pts = std::count(d.inner_point.begin(), d.inner_point.end(), true) +
                         std::count(d.inner_grad_point.begin(), d.inner_grad_point.end(), true);
      if (n_inner_pts == 0) {
        it = domains_.erase(it);
      } else {
        ++it;
      }
    }
  }

  Bbox domain_bbox(const Domain& domain) const {
    auto points = points_(domain.point_indices, Eigen::all);
    auto grad_points = grad_points_(domain.grad_point_indices, Eigen::all);

    return Bbox::from_points(points).convex_hull(Bbox::from_points(grad_points));
  }

  static double round_half_to_even(double d) {
    return std::ceil((d - 0.5) / 2.0) + std::floor((d + 0.5) / 2.0);
  }

  const Points points_;
  const Points grad_points_;
  const std::vector<index_t> poly_point_idcs_;
  std::list<Domain> domains_;
};

}  // namespace polatory::preconditioner
