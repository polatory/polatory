#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <polatory/geometry/point3d.hpp>
#include <polatory/isosurface/edge.hpp>
#include <polatory/isosurface/predicates.hpp>
#include <polatory/isosurface/types.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace polatory::isosurface::snapper {

using geometry::Point2;
using geometry::Points2;

// A constrained Delaunay triangulation of a simple polygon with interior points. The
// triangulation is computed on construction by ear clipping (an initial triangulation),
// inserting the interior points, then Lawson flips (the Delaunay property); the result is
// read back with faces().
class Triangulation {
 public:
  // boundary  the polygon vertices in order (CW or CCW; the orientation is detected).
  //           Consecutive vertices are constraint edges and are preserved.
  // interior  points that lie strictly inside the polygon.
  //
  // boundary_edges, if given, labels each boundary vertex with the original edge(s) it lies
  // on (an original vertex joins two; a vertex inserted on an edge lies on one; -1 pads the
  // unused slot). No triangle is then allowed to connect two vertices sharing an edge label, which
  // would be a diagonal running along that subdivided edge. This is what makes two patches
  // meeting at a shared edge agree on its subdivision (a manifold seam): each must fan its
  // sub-edges inward to its own interior rather than cut across them.
  // faces_ preallocated to its upper bound nb_ - 2 + 2 * ni_, filled via nf_, then trimmed.
  Triangulation(const std::vector<Point2>& boundary, const std::vector<Point2>& interior,
                std::vector<std::array<int, 2>> boundary_edges = {})
      : nb_(check_nb(static_cast<Index>(boundary.size()))),
        ni_(static_cast<Index>(interior.size())),
        boundary_edges_(std::move(boundary_edges)),
        points_(nb_ + ni_, 2),
        faces_(nb_ - 2 + 2 * ni_, 3) {
    for (Index i = 0; i < nb_; i++) {
      points_.row(i) = boundary.at(i);
    }
    for (Index i = 0; i < ni_; i++) {
      points_.row(nb_ + i) = interior.at(i);
    }

    // A length scale for the tolerances.
    Point2 lo = points_.colwise().minCoeff();
    Point2 hi = points_.colwise().maxCoeff();
    scale_ = (hi - lo).norm();

    // The constraint edges are the polygon boundary edges (undirected, so independent of
    // the winding); kept sorted so membership is a binary search.
    constraints_.reserve(nb_);
    for (Index i = 0; i < nb_; i++) {
      constraints_.push_back({i, (i + 1) % nb_});
    }
    std::ranges::sort(constraints_);

    ear_clip();
    insert_interior();
    make_delaunay();

    faces_.conservativeResize(nf_, Eigen::NoChange);  // trim the unused upper-bound rows
  }

  // CCW triangles as index triples into the concatenation {boundary..., interior...} (so an
  // index < boundary.size() refers to boundary[index], otherwise to interior[index -
  // boundary.size()]). The triangles cover the polygon without overlap and every boundary
  // edge appears.
  const Faces& faces() const { return faces_; }

  // False if the boundary polygon was found not to be simple, in which case the result is
  // an unreliable fan triangulation and the caller should treat the input as invalid.
  bool simple() const { return simple_; }

 private:
  void add_face(const Face& f) { faces_.row(nf_++) = f; }

  static Index check_nb(Index nb) {
    if (nb < 3) {
      throw std::invalid_argument("triangulation needs at least 3 boundary vertices");
    }
    return nb;
  }

  // Triangulate the boundary polygon by ear clipping.
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
        add_face({prev, cur, next});
        ring.erase(ring.begin() + i);
        clipped = true;
        break;
      }
      if (!clipped) {
        // The polygon is not simple; fan-triangulate the remainder so we terminate, and
        // report the failure.
        simple_ = false;
        for (Index i = 1; i + 1 < static_cast<Index>(ring.size()); i++) {
          add_face({ring.at(0), ring.at(i), ring.at(i + 1)});
        }
        ring.clear();
        break;
      }
    }
    if (ring.size() == 3) {
      add_face({ring.at(0), ring.at(1), ring.at(2)});
    }
  }

  // Whether x lies in the CCW triangle (a, b, c), including its boundary.
  static bool in_triangle(const Point2& x, const Point2& a, const Point2& b, const Point2& c) {
    return orient2d(a, b, x) >= 0.0 && orient2d(b, c, x) >= 0.0 && orient2d(c, a, x) >= 0.0;
  }

  // Insert the interior points. A point strictly inside a triangle splits it (1 -> 3); a
  // point on an interior edge splits both triangles sharing it (2 -> 4) to avoid degenerate
  // triangles; a point on a vertex or a constraint edge is dropped (splitting a constraint
  // edge would desynchronize the shared boundary).
  void insert_interior() {
    auto n = static_cast<Index>(points_.rows());
    for (Index v = nb_; v < n; v++) {
      Index best = -1;
      auto best_min = -std::numeric_limits<double>::infinity();
      std::array<double, 3> bl{};
      for (Index f = 0; f < nf_; f++) {
        Face face = faces_.row(f);
        auto a = orient2d(points_.row(face(0)), points_.row(face(1)), points_.row(face(2)));
        if (!(a > 0.0)) {
          continue;
        }
        std::array<double, 3> l{
            orient2d(points_.row(v), points_.row(face(1)), points_.row(face(2))) / a,
            orient2d(points_.row(face(0)), points_.row(v), points_.row(face(2))) / a,
            orient2d(points_.row(face(0)), points_.row(face(1)), points_.row(v)) / a};
        auto mn = std::min({l[0], l[1], l[2]});
        if (mn > best_min) {
          best_min = mn;
          best = f;
          bl = l;
        }
      }
      if (best < 0 || best_min < -1e-9) {
        continue;  // not inside any triangle (should not happen for interior points)
      }

      constexpr double kOnEdge = 1e-9;
      if (best_min > kOnEdge) {
        Face face = faces_.row(best);
        faces_.row(best) = Face(face(0), face(1), v);
        add_face({face(1), face(2), v});
        add_face({face(2), face(0), v});
        continue;
      }

      // The point is on the edge opposite the smallest-barycentric vertex.
      auto kmin = static_cast<int>(std::ranges::min_element(bl) - bl.begin());
      if (bl.at((kmin + 1) % 3) < kOnEdge || bl.at((kmin + 2) % 3) < kOnEdge) {
        continue;  // coincides with an existing vertex
      }
      auto u = faces_(best, (kmin + 1) % 3);
      auto w = faces_(best, (kmin + 2) % 3);
      if (is_constraint({u, w})) {
        continue;  // never split a boundary edge
      }

      std::vector<Index> incident;
      for (Index f = 0; f < nf_; f++) {
        Face face = faces_.row(f);
        bool has_u = face(0) == u || face(1) == u || face(2) == u;
        bool has_w = face(0) == w || face(1) == w || face(2) == w;
        if (has_u && has_w) {
          incident.push_back(f);
        }
      }
      for (auto f : incident) {
        Face face = faces_.row(f);
        auto i = 0;
        for (auto k = 0; k < 3; k++) {
          if ((face(k) == u && face((k + 1) % 3) == w) ||
              (face(k) == w && face((k + 1) % 3) == u)) {
            i = k;
            break;
          }
        }
        faces_.row(f) = Face(face(i), v, face((i + 2) % 3));
        add_face({v, face((i + 1) % 3), face((i + 2) % 3)});
      }
    }
  }

  bool is_constraint(const Edge& e) const { return std::ranges::binary_search(constraints_, e); }

  // Restore the (constrained) Delaunay property by Lawson flips, never flipping a
  // constraint edge or creating an along-edge chord. Each pass collects the triangles
  // sharing each edge, then flips every non-Delaunay interior edge whose two triangles have
  // not already been flipped in that pass; it repeats until a pass changes nothing. Flipping
  // all such independent edges per pass — rather than one, then rebuilding — keeps the
  // number of rebuilds proportional to the number of passes, which is small.
  void make_delaunay() {
    auto s2 = scale_ * scale_;
    auto incircle_tol = 1e-10 * s2 * s2;  // incircle scales like length^4
    auto area_tol = 1e-12 * s2;           // orient scales like length^2

    auto budget = 10 + 3 * static_cast<long long>(nf_);
    bool changed = true;
    while (changed && budget-- > 0) {
      changed = false;

      // (edge, face) for every directed edge, sorted so an interior edge's two faces
      // are adjacent in the list.
      std::vector<std::pair<Edge, Index>> edge_faces;
      edge_faces.reserve(3 * nf_);
      for (Index f = 0; f < nf_; f++) {
        Face face = faces_.row(f);
        for (auto k = 0; k < 3; k++) {
          edge_faces.emplace_back(Edge{face(k), face((k + 1) % 3)}, f);
        }
      }
      std::ranges::sort(edge_faces);

      std::vector<bool> flipped(nf_, false);
      // After the sort, all entries for one undirected edge are adjacent, so walk the list in
      // runs [begin, end) of equal edges. Each incident face contributes one entry, so the run
      // length is the number of faces sharing the edge: 2 for an interior edge (the only kind
      // that can be flipped), 1 for a boundary edge, more than 2 for a non-manifold edge.
      for (auto first = edge_faces.cbegin(), last = first; first != edge_faces.cend();
           first = last) {
        auto e = first->first;
        last = std::ranges::find_if(first, edge_faces.cend(),
                                    [&](const auto& ef) { return ef.first != e; });
        if (std::distance(first, last) != 2) {
          continue;
        }
        auto f0 = first->second;
        auto f1 = (first + 1)->second;
        if (flipped.at(f0) || flipped.at(f1) || is_constraint(e)) {
          continue;
        }

        // face0 = (x, y, c) with directed edge x->y equal to e; d is face1's apex.
        Face face0 = faces_.row(f0);
        Face face1 = faces_.row(f1);
        auto i0 = 0;
        for (auto k = 0; k < 3; k++) {
          if (e == Edge{face0(k), face0((k + 1) % 3)}) {
            i0 = k;
            break;
          }
        }
        auto x = face0(i0);
        auto y = face0((i0 + 1) % 3);
        auto c = face0((i0 + 2) % 3);
        auto [ea, eb] = e;
        auto d = face1(0);
        for (auto v : face1) {
          if (v != ea && v != eb) {
            d = v;
          }
        }

        // Never flip to a diagonal that runs along a subdivided edge (the two patches
        // sharing that edge could not agree on it).
        if (shares_edge({c, d})) {
          continue;
        }

        // Flip only when d is decisively inside the circumcircle of (x, y, c) and both
        // resulting triangles are strictly positive.
        if (!(incircle(points_.row(x), points_.row(y), points_.row(c), points_.row(d)) >
              incircle_tol)) {
          continue;
        }
        if (!(orient2d(points_.row(x), points_.row(d), points_.row(c)) > area_tol &&
              orient2d(points_.row(d), points_.row(y), points_.row(c)) > area_tol)) {
          continue;
        }

        faces_.row(f0) = Face(x, d, c);
        faces_.row(f1) = Face(d, y, c);
        flipped.at(f0) = true;
        flipped.at(f1) = true;
        changed = true;
      }
    }
  }

  // Whether the edge's two boundary vertices lie on a common original edge (see the
  // constructor): a diagonal between them would run along a subdivided edge and must never
  // be created.
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
  Index nf_{};                                      // number of faces filled into faces_
  double scale_{};                                  // length scale for tolerances
  std::vector<std::array<int, 2>> boundary_edges_;  // original-edge labels per boundary vertex
  std::vector<Edge> constraints_;                   // boundary edges, sorted
  Points2 points_;                                  // boundary..., then interior...
  Faces faces_;
  bool simple_{true};
};

}  // namespace polatory::isosurface::snapper
