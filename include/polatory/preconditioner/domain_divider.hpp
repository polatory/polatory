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
class DomainDivider {
  static constexpr int kDim = Dim;
  using Bbox = geometry::Bbox<kDim>;
  using Domain = Domain<kDim>;
  using Point = geometry::Point<kDim>;
  using Points = geometry::Points<kDim>;

  static constexpr double kOverlapQuota = 0.5;
  static constexpr Index kMaxLeafSize = 1024;

 public:
  template <class DerivedPoints, class DerivedGradPoints>
  DomainDivider(const Eigen::MatrixBase<DerivedPoints>& points,
                const Eigen::MatrixBase<DerivedGradPoints>& grad_points,
                const std::vector<Index>& point_indices,
                const std::vector<Index>& grad_point_indices,
                const std::vector<Index>& poly_point_indices)
      : points_(points), grad_points_(grad_points), poly_point_idcs_(poly_point_indices) {
    Domain root;

    root.point_indices = point_indices;
    root.grad_point_indices = grad_point_indices;

    root.inner_point = std::vector<bool>(point_indices.size(), true);
    root.inner_grad_point = std::vector<bool>(grad_point_indices.size(), true);

    domains_.push_back(std::move(root));

    divide_domains();
  }

  std::pair<std::vector<Index>, std::vector<Index>> choose_coarse_points(double ratio) const {
    std::vector<Index> idcs(poly_point_idcs_);
    std::vector<Index> grad_idcs;

    std::mt19937 gen;

    auto n_poly_points = static_cast<Index>(poly_point_idcs_.size());
    for (const auto& d : domains_) {
      auto mu = d.num_points();
      auto sigma = d.num_grad_points();

      std::vector<MixedPoint> mixed_points;
      for (Index i = n_poly_points; i < mu; i++) {
        if (d.inner_point.at(i)) {
          mixed_points.emplace_back(d.point_indices.at(i), true, false);
        }
      }
      for (Index i = 0; i < sigma; i++) {
        if (d.inner_grad_point.at(i)) {
          mixed_points.emplace_back(d.grad_point_indices.at(i), true, true);
        }
      }
      std::shuffle(mixed_points.begin(), mixed_points.end(), gen);

      auto n_coarse_points =
          static_cast<Index>(round_half_to_even(ratio * static_cast<double>(mixed_points.size())));

      Index count{};
      for (const auto& p : mixed_points) {
        if (count == n_coarse_points) {
          break;
        }

        (p.grad ? grad_idcs : idcs).push_back(p.index);
        count++;
      }
    }

    return {std::move(idcs), std::move(grad_idcs)};
  }

  const std::list<Domain>& domains() const { return domains_; }

  std::list<Domain> into_domains() { return std::move(domains_); }

 private:
  struct MixedPoint {
    Index index{};
    bool inner{};
    bool grad{};

    int multiplicity() const { return grad ? kDim : 1; }

    Point point(const Points& points, const Points& grad_points) const {
      return grad ? grad_points.row(index) : points.row(index);
    }
  };

  void divide_domain(typename std::list<Domain>::iterator it) {
    auto& d = *it;
    auto mu = d.num_points();
    auto sigma = d.num_grad_points();

    std::vector<MixedPoint> mixed_points;
    for (Index i = 0; i < mu; i++) {
      mixed_points.emplace_back(d.point_indices.at(i), d.inner_point.at(i), false);
    }
    for (Index i = 0; i < sigma; i++) {
      mixed_points.emplace_back(d.grad_point_indices.at(i), d.inner_grad_point.at(i), true);
    }

    // If the points are axis-aligned, it is important to sort them
    // not only along the longest axis but also along the other axes.
    auto width = domain_bbox(d).width();
    std::array<int, kDim> axes;
    std::iota(axes.begin(), axes.end(), 0);
    std::sort(axes.begin(), axes.end(), [&width](auto i, auto j) { return width(i) > width(j); });
    std::sort(mixed_points.begin(), mixed_points.end(),
              [this, &axes](const auto& a, const auto& b) {
                auto p = a.point(points_, grad_points_);
                auto q = b.point(points_, grad_points_);
                for (auto axis : axes) {
                  if (p(axis) != q(axis)) {
                    return p(axis) < q(axis);
                  }
                }
                return false;
              });

    std::vector<Index> prefix_sum_mult{0};
    for (const auto& p : mixed_points) {
      prefix_sum_mult.push_back(prefix_sum_mult.back() + p.multiplicity());
    }

    auto n_points_mult = mu + kDim * sigma;
    auto q = kOverlapQuota * static_cast<double>(kMaxLeafSize) / static_cast<double>(n_points_mult);
    auto n_subdomain_points_mult = static_cast<Index>(
        round_half_to_even((1.0 + q) / 2.0 * static_cast<double>(n_points_mult)));
    auto left_partition_mult = n_points_mult - n_subdomain_points_mult;
    auto right_partition_mult = n_subdomain_points_mult;
    auto mid_mult = static_cast<Index>(
        round_half_to_even(static_cast<double>(left_partition_mult + right_partition_mult) / 2.0));

    auto n_points = mu + sigma;
    auto left_partition = static_cast<Index>(std::distance(
        prefix_sum_mult.begin(),
        std::upper_bound(prefix_sum_mult.begin(), prefix_sum_mult.end(), left_partition_mult) - 1));
    auto right_partition = static_cast<Index>(std::distance(
        prefix_sum_mult.begin(),
        std::upper_bound(prefix_sum_mult.begin(), prefix_sum_mult.end(), right_partition_mult) -
            1));
    auto mid = static_cast<Index>(std::distance(
        prefix_sum_mult.begin(),
        std::upper_bound(prefix_sum_mult.begin(), prefix_sum_mult.end(), mid_mult) - 1));

    Domain left;
    Domain right;

    for (Index i = 0; i < right_partition; i++) {
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

    for (Index i = left_partition; i < n_points; i++) {
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
      if (d.num_points() + kDim * d.num_grad_points() <= kMaxLeafSize) {
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
  const std::vector<Index> poly_point_idcs_;
  std::list<Domain> domains_;
};

}  // namespace polatory::preconditioner
