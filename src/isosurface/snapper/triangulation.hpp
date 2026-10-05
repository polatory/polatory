#pragma once

#include <algorithm>
#include <array>
#include <jizai/geometry/point3d.hpp>
#include <jizai/isosurface/edge.hpp>
#include <jizai/isosurface/predicates.hpp>
#include <jizai/isosurface/types.hpp>
#include <jizai/types.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../abstract_mesh.hpp"

namespace jizai::isosurface::snapper {

using geometry::Point2;
using geometry::Points2;

// A constrained Delaunay triangulation of a simple polygon with interior points.
class Triangulation {
 public:
  // No diagonal joins two boundary vertices that share a label (-1 = no label).
  Triangulation(const std::vector<Point2>& boundary, const std::vector<Point2>& interior,
                std::vector<std::array<int, 2>> boundary_labels = {})
      : nb_(check_nb(static_cast<Index>(boundary.size()))),
        ni_(static_cast<Index>(interior.size())),
        boundary_labels_(std::move(boundary_labels)),
        points_(nb_ + ni_, 2),
        mesh_(nb_ - 2 + 2 * ni_) {
    for (Index i = 0; i < nb_; i++) {
      points_.row(i) = boundary.at(i);
    }
    for (Index i = 0; i < ni_; i++) {
      points_.row(nb_ + i) = interior.at(i);
    }

    Point2 lo = points_.colwise().minCoeff();
    Point2 hi = points_.colwise().maxCoeff();
    scale_ = (hi - lo).norm();

    constraints_.reserve(nb_);
    for (Index i = 0; i < nb_; i++) {
      constraints_.push_back({i, (i + 1) % nb_});
    }
    std::ranges::sort(constraints_);

    ear_clip();
    if (!simple_) {
      return;
    }

    insert_interior();
    make_delaunay();

    faces_ = std::move(mesh_).take_faces();
  }

  // CCW triangles, indexing the boundary points followed by the interior points.
  const Faces& faces() const { return faces_; }

  bool simple() const { return simple_; }

 private:
  static Index check_nb(Index nb) {
    if (nb < 3) {
      throw std::invalid_argument("triangulation needs at least 3 boundary vertices");
    }
    return nb;
  }

  void ear_clip() {
    std::vector<Index> ring(nb_);
    for (Index i = 0; i < nb_; i++) {
      ring.at(i) = i;
    }
    double signed_area2 = 0.0;
    for (Index i = 0; i < nb_; i++) {
      const auto& a = points_.row(ring.at(i));
      const auto& b = points_.row(ring.at((i + 1) % nb_));
      signed_area2 += a(0) * b(1) - b(0) * a(1);
    }
    if (signed_area2 < 0.0) {
      std::ranges::reverse(ring);
    }

    auto area_tol = 1e-14 * scale_ * scale_;
    while (static_cast<Index>(ring.size()) > 3) {
      auto m = static_cast<Index>(ring.size());
      bool clipped = false;
      for (Index i = 0; i < m; i++) {
        auto cur = ring.at(i);
        auto prev = ring.at((i + m - 1) % m);
        auto next = ring.at((i + 1) % m);
        if (!(orient2d(points_.row(prev), points_.row(cur), points_.row(next)) > area_tol)) {
          continue;
        }
        if (shares_label({prev, next})) {
          continue;
        }
        bool ear = true;
        for (Index j = 0; j < m; j++) {
          auto vj = ring.at(j);
          if (vj == prev || vj == cur || vj == next) {
            continue;
          }
          if (in_triangle(points_.row(vj), points_.row(prev), points_.row(cur),
                          points_.row(next))) {
            ear = false;
            break;
          }
        }
        if (!ear) {
          continue;
        }
        mesh_.add_face({prev, cur, next});
        ring.erase(ring.begin() + i);
        clipped = true;
        break;
      }
      if (!clipped) {
        simple_ = false;
        return;
      }
    }
    mesh_.add_face({ring.at(0), ring.at(1), ring.at(2)});
  }

  static bool in_triangle(const Point2& x, const Point2& a, const Point2& b, const Point2& c) {
    return orient2d(a, b, x) >= 0.0 && orient2d(b, c, x) >= 0.0 && orient2d(c, a, x) >= 0.0;
  }

  void insert_interior() {
    auto n = static_cast<Index>(points_.rows());
    for (Index v = nb_; v < n; v++) {
      Index best = -1;
      auto best_min = -std::numeric_limits<double>::infinity();
      std::array<double, 3> bl{};
      auto nf = mesh_.num_faces();
      for (Index fi = 0; fi < nf; fi++) {
        auto f = mesh_.face(fi);
        auto a = orient2d(points_.row(f(0)), points_.row(f(1)), points_.row(f(2)));
        if (!(a > 0.0)) {
          continue;
        }
        std::array<double, 3> l{orient2d(points_.row(v), points_.row(f(1)), points_.row(f(2))) / a,
                                orient2d(points_.row(f(0)), points_.row(v), points_.row(f(2))) / a,
                                orient2d(points_.row(f(0)), points_.row(f(1)), points_.row(v)) / a};
        auto mn = std::min({l[0], l[1], l[2]});
        if (mn > best_min) {
          best_min = mn;
          best = fi;
          bl = l;
        }
      }
      if (best < 0 || best_min < -1e-9) {
        continue;  // should not happen
      }

      constexpr double kOnEdge = 1e-9;
      if (best_min > kOnEdge) {
        mesh_.insert_in_face(best, v);
        continue;
      }

      // The point is on the edge opposite the smallest-barycentric vertex.
      auto kmin = static_cast<int>(std::ranges::min_element(bl) - bl.begin());
      if (bl.at((kmin + 1) % 3) < kOnEdge || bl.at((kmin + 2) % 3) < kOnEdge) {
        continue;  // coincides with an existing vertex
      }
      auto bf = mesh_.face(best);
      auto u = bf((kmin + 1) % 3);
      auto w = bf((kmin + 2) % 3);
      if (is_constraint({u, w})) {
        continue;
      }
      mesh_.insert_on_edge({u, w}, v);
    }
  }

  bool is_constraint(const Edge& e) const { return std::ranges::binary_search(constraints_, e); }

  void make_delaunay() {
    auto s2 = scale_ * scale_;
    auto incircle_tol = 1e-10 * s2 * s2;
    auto area_tol = 1e-12 * s2;

    auto budget = 10 + 3 * static_cast<long long>(mesh_.num_faces());
    bool changed = true;
    while (changed && budget-- > 0) {
      changed = false;

      std::vector<Halfedge> hs;
      mesh_.for_each_halfedge([&](Halfedge h) {
        if (mesh_.from(h) < mesh_.to(h) && mesh_.opposite(h).is_valid()) {
          hs.push_back(h);
        }
      });
      std::ranges::sort(hs, [&](Halfedge a, Halfedge b) {
        return std::pair(mesh_.from(a), mesh_.to(a)) < std::pair(mesh_.from(b), mesh_.to(b));
      });

      std::vector<bool> flipped(mesh_.num_faces(), false);
      for (auto h : hs) {
        Edge e{mesh_.from(h), mesh_.to(h)};
        if (is_constraint(e)) {
          continue;
        }
        auto opp_h = mesh_.opposite(h);
        auto fi0 = mesh_.face(h);
        auto fi1 = mesh_.face(opp_h);
        if (fi0 < 0 || fi1 < 0 || flipped.at(fi0) || flipped.at(fi1)) {
          continue;
        }

        auto x = mesh_.from(h);
        auto y = mesh_.to(h);
        auto c = mesh_.apex(h);
        auto d = mesh_.apex(opp_h);

        if (shares_label({c, d})) {
          continue;
        }
        if (!(incircle(points_.row(x), points_.row(y), points_.row(c), points_.row(d)) >
              incircle_tol)) {
          continue;
        }
        if (!(orient2d(points_.row(x), points_.row(d), points_.row(c)) > area_tol &&
              orient2d(points_.row(d), points_.row(y), points_.row(c)) > area_tol)) {
          continue;
        }

        if (mesh_.has_edge({c, d})) {
          continue;  // possible with inexact predicates
        }

        mesh_.flip(e);
        flipped.at(fi0) = true;
        flipped.at(fi1) = true;
        changed = true;
      }
    }
  }

  bool shares_label(const Edge& e) const {
    auto [i, j] = e;
    if (boundary_labels_.empty() || i >= nb_ || j >= nb_) {
      return false;
    }
    for (auto a : boundary_labels_.at(i)) {
      if (a < 0) {
        continue;
      }
      for (auto b : boundary_labels_.at(j)) {
        if (a == b) {
          return true;
        }
      }
    }
    return false;
  }

  Index nb_{};
  Index ni_{};
  double scale_{};
  std::vector<std::array<int, 2>> boundary_labels_;
  std::vector<Edge> constraints_;
  Points2 points_;
  AbstractMesh mesh_;
  Faces faces_;
  bool simple_{true};
};

}  // namespace jizai::isosurface::snapper
