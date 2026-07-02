#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/predicates.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "abstract_mesh.hpp"

namespace polatory::isosurface::snapper {

using geometry::Point2;
using geometry::Points2;

// A constrained Delaunay triangulation of a simple polygon with interior points, built on
// construction (ear clip, insert interior, Lawson flips) and read with faces().
class Triangulation {
 public:
  // boundary: polygon vertices in order (CW/CCW auto-detected), consecutive pairs constraint edges.
  // interior: points strictly inside. boundary_edges (optional): each boundary vertex's
  // original-edge label(s); no triangle joins two vertices sharing a label, so patches meeting at a
  // shared edge agree on its subdivision (a manifold seam) rather than cutting a diagonal along it.
  Triangulation(const std::vector<Point2>& boundary, const std::vector<Point2>& interior,
                std::vector<std::array<int, 2>> boundary_edges = {})
      : nb_(check_nb(static_cast<Index>(boundary.size()))),
        ni_(static_cast<Index>(interior.size())),
        boundary_edges_(std::move(boundary_edges)),
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

    // Boundary edges, sorted for binary-search membership.
    constraints_.reserve(nb_);
    for (Index i = 0; i < nb_; i++) {
      constraints_.push_back({i, (i + 1) % nb_});
    }
    std::ranges::sort(constraints_);

    try {
      ear_clip();
      insert_interior();
      make_delaunay();
    } catch (const std::runtime_error&) {
      // A non-simple polygon drove the triangulation non-manifold; the result is unreliable.
      simple_ = false;
    }

    faces_ = std::move(mesh_).take_faces();
  }

  // CCW triangles, indexed into {boundary..., interior...} (index < boundary.size() is a boundary
  // vertex, else interior[index - boundary.size()]).
  const Faces& faces() const { return faces_; }

  // False if the polygon was not simple; the result is then an unreliable fan, treat input as
  // invalid.
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
          continue;  // reflex or flat: not an ear
        }
        if (shares_edge({prev, next})) {
          continue;  // the chord prev-next would cut along a subdivided edge across cur
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
        // Not simple: fan-triangulate the remainder to terminate, and flag the failure.
        simple_ = false;
        for (Index i = 1; i + 1 < static_cast<Index>(ring.size()); i++) {
          mesh_.add_face({ring.at(0), ring.at(i), ring.at(i + 1)});
        }
        ring.clear();
        break;
      }
    }
    if (ring.size() == 3) {
      mesh_.add_face({ring.at(0), ring.at(1), ring.at(2)});
    }
  }

  static bool in_triangle(const Point2& x, const Point2& a, const Point2& b, const Point2& c) {
    return orient2d(a, b, x) >= 0.0 && orient2d(b, c, x) >= 0.0 && orient2d(c, a, x) >= 0.0;
  }

  // A point inside a triangle splits it (1 -> 3); on an interior edge, both its faces (2 -> 4); on
  // a vertex or constraint edge, it is dropped (splitting a constraint edge would desync the
  // boundary).
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
        continue;  // not inside any triangle (should not happen for interior points)
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
        continue;  // never split a boundary edge
      }
      mesh_.insert_on_edge({u, w}, v);
    }
  }

  bool is_constraint(const Edge& e) const { return std::ranges::binary_search(constraints_, e); }

  // Lawson flips to (constrained) Delaunay: each pass flips every non-Delaunay, non-constraint
  // interior edge whose faces are not yet flipped this pass, in sorted-edge order, until stable.
  void make_delaunay() {
    auto s2 = scale_ * scale_;
    auto incircle_tol = 1e-10 * s2 * s2;  // incircle scales like length^4
    auto area_tol = 1e-12 * s2;           // orient scales like length^2

    auto budget = 10 + 3 * static_cast<long long>(mesh_.num_faces());
    bool changed = true;
    while (changed && budget-- > 0) {
      changed = false;

      std::vector<Edge> edges;
      mesh_.for_each_edge([&](const Edge& e, const auto& sides) {
        if (sides[0] >= 0 && sides[1] >= 0) {
          edges.push_back(e);
        }
      });
      std::ranges::sort(edges);

      std::vector<bool> flipped(mesh_.num_faces(), false);
      for (const auto& e : edges) {
        if (is_constraint(e)) {
          continue;
        }
        auto sides = mesh_.faces_of(e);
        if (sides.size() < 2 || flipped.at(sides[0]) || flipped.at(sides[1])) {
          continue;  // a boundary edge now, or a face already flipped this pass
        }
        auto fi0 = sides[0];
        auto fi1 = sides[1];

        // fi0 traverses e.a -> e.b, so (e.a, e.b, c) is CCW; d is fi1's apex.
        auto x = e.a;
        auto y = e.b;
        auto c = AbstractMesh::opposite(mesh_.face(fi0), e);
        auto d = AbstractMesh::opposite(mesh_.face(fi1), e);

        if (shares_edge({c, d})) {
          continue;  // never cut a diagonal along a subdivided edge (the patches would disagree)
        }

        // Flip only when d is decisively inside (x, y, c)'s circumcircle and both new triangles are
        // strictly positive.
        if (!(incircle(points_.row(x), points_.row(y), points_.row(c), points_.row(d)) >
              incircle_tol)) {
          continue;
        }
        if (!(orient2d(points_.row(x), points_.row(d), points_.row(c)) > area_tol &&
              orient2d(points_.row(d), points_.row(y), points_.row(c)) > area_tol)) {
          continue;
        }

        mesh_.flip(e);
        flipped.at(fi0) = true;
        flipped.at(fi1) = true;
        changed = true;
      }
    }
  }

  // True if e's endpoints share an original edge label: a diagonal along it would desync the
  // patches.
  bool shares_edge(const Edge& e) const {
    auto [i, j] = e;
    if (boundary_edges_.empty() || i >= nb_ || j >= nb_) {
      return false;
    }
    for (auto a : boundary_edges_.at(i)) {
      if (a < 0) {
        continue;
      }
      for (auto b : boundary_edges_.at(j)) {
        if (a == b) {
          return true;
        }
      }
    }
    return false;
  }

  Index nb_{};                                      // number of boundary vertices
  Index ni_{};                                      // number of interior points
  double scale_{};                                  // length scale for tolerances
  std::vector<std::array<int, 2>> boundary_edges_;  // original-edge labels per boundary vertex
  std::vector<Edge> constraints_;                   // boundary edges, sorted
  Points2 points_;                                  // boundary..., then interior...
  AbstractMesh mesh_;
  Faces faces_;
  bool simple_{true};
};

}  // namespace polatory::isosurface::snapper
