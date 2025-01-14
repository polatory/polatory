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
#include <queue>
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
      : points_(points),
        grad_points_(grad_points),
        point_idcs_(point_indices),
        grad_point_idcs_(grad_point_indices),
        poly_point_idcs_(poly_point_indices) {
    Domain root;

    root.point_indices = point_indices;
    root.grad_point_indices = grad_point_indices;

    root.inner_point = std::vector<bool>(point_indices.size(), true);
    root.inner_grad_point = std::vector<bool>(grad_point_indices.size(), true);

    domains_.push_back(std::move(root));

    divide_domains();
  }

  std::pair<std::vector<Index>, std::vector<Index>> choose_coarse_points(
      Index n_coarse_points) const {
    std::vector<Index> idcs(poly_point_idcs_);
    std::vector<Index> grad_idcs;

    std::priority_queue<Cluster> clusters;
    std::vector<MixedPoint> root_points;
    for (auto i : point_idcs_) {
      if (std::find(poly_point_idcs_.begin(), poly_point_idcs_.end(), i) ==
          poly_point_idcs_.end()) {
        root_points.emplace_back(i, true, false);
      }
    }
    for (auto i : grad_point_idcs_) {
      root_points.emplace_back(i, true, true);
    }
    Cluster root_cluster(std::move(root_points), 0);
    initialize_cluster(root_cluster);
    auto current_size = root_cluster.center.multiplicity();
    clusters.push(root_cluster);

    while (current_size < n_coarse_points) {
      const auto& cluster = clusters.top();
      const auto& points = cluster.points;

      std::vector<Index> prefix_sum_mult{0};
      for (const auto& p : points) {
        prefix_sum_mult.push_back(prefix_sum_mult.back() + p.multiplicity());
      }
      auto cluster_size = prefix_sum_mult.back();
      auto mid_mult =
          static_cast<Index>(round_half_to_even(static_cast<double>(cluster_size) / 2.0));
      auto mid = static_cast<Index>(std::distance(
          prefix_sum_mult.begin(),
          std::upper_bound(prefix_sum_mult.begin(), prefix_sum_mult.end(), mid_mult) - 1));

      std::vector<MixedPoint> left_points(points.begin(), points.begin() + mid);
      std::vector<MixedPoint> right_points(points.begin() + mid, points.end());

      Cluster left(std::move(left_points), cluster.level + 1);
      Cluster right(std::move(right_points), cluster.level + 1);
      initialize_cluster(left);
      initialize_cluster(right);

      current_size -= cluster.center.multiplicity();
      current_size += left.center.multiplicity();
      current_size += right.center.multiplicity();

      clusters.pop();
      clusters.push(std::move(left));
      clusters.push(std::move(right));
    }

    while (!clusters.empty()) {
      const auto& cluster = clusters.top();
      const auto& p = cluster.center;
      (p.grad ? grad_idcs : idcs).push_back(p.index);
      clusters.pop();
    }

    return {std::move(idcs), std::move(grad_idcs)};
  }

  const std::list<Domain>& domains() const { return domains_; }

  std::list<Domain> into_domains() && { return std::move(domains_); }

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

  struct Cluster {
    Cluster(std::vector<MixedPoint> points, int level) : level(level), points(std::move(points)) {}

    bool operator<(const Cluster& other) const {
      if (level != other.level) {
        return level > other.level;
      }

      return bbox.width().prod() < other.bbox.width().prod();
    }

    int level{};
    std::vector<MixedPoint> points;
    Bbox bbox;
    MixedPoint center;
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

  void initialize_cluster(Cluster& cluster) const {
    auto& points = cluster.points;

    Bbox bbox{};
    for (const auto& a : points) {
      Point p = a.point(points_, grad_points_);
      bbox = bbox.convex_hull(Bbox{p, p});
    }
    cluster.bbox = bbox;

    cluster.center = *std::min_element(
        points.begin(), points.end(), [this, &bbox](const auto& a, const auto& b) {
          auto p = a.point(points_, grad_points_);
          auto q = b.point(points_, grad_points_);
          return (p - bbox.center()).squaredNorm() < (q - bbox.center()).squaredNorm();
        });

    auto width = bbox.width();
    std::array<int, kDim> axes;
    std::iota(axes.begin(), axes.end(), 0);
    std::sort(axes.begin(), axes.end(), [&width](auto i, auto j) { return width(i) > width(j); });
    std::sort(points.begin(), points.end(), [this, &axes](const auto& a, const auto& b) {
      auto p = a.point(points_, grad_points_);
      auto q = b.point(points_, grad_points_);
      for (auto axis : axes) {
        if (p(axis) != q(axis)) {
          return p(axis) < q(axis);
        }
      }
      return false;
    });
  }

  static double round_half_to_even(double d) {
    return std::ceil((d - 0.5) / 2.0) + std::floor((d + 0.5) / 2.0);
  }

  const Points points_;
  const Points grad_points_;
  const std::vector<Index> point_idcs_;
  const std::vector<Index> grad_point_idcs_;
  const std::vector<Index> poly_point_idcs_;
  std::list<Domain> domains_;
};

}  // namespace polatory::preconditioner
