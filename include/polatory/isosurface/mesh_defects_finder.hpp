// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "polatory/common/utility.hpp"
#include "types.hpp"

namespace polatory {
namespace isosurface {

class mesh_defects_finder {
  typedef std::pair<vertex_index, vertex_index> halfedge;

  typedef int face_index;
  typedef std::pair<face_index, bool> face_index_bool;
  typedef std::vector<face_index_bool> face_index_bools;

  const std::vector<Eigen::Vector3d>& vertices;
  const std::vector<face>& faces;

  static halfedge opposite_halfedge(halfedge e) {
    return std::make_pair(e.second, e.first);
  }

  halfedge vertex_outgoing_halfedge(face_index fi, const vertex_index vi) const {
    const face& face = faces[fi];
    if (face[0] == vi) return std::make_pair(vi, face[1]);
    if (face[1] == vi) return std::make_pair(vi, face[2]);
    assert(face[2] == vi);
    return std::make_pair(vi, face[0]);
  }

  halfedge vertex_incoming_halfedge(face_index fi, const vertex_index vi) const {
    const face& face = faces[fi];
    if (face[0] == vi) return std::make_pair(face[2], vi);
    if (face[1] == vi) return std::make_pair(face[0], vi);
    assert(face[2] == vi);
    return std::make_pair(face[1], vi);
  }

  face_index_bools::iterator halfedge_face(face_index_bools& fi_bools, halfedge he) const {
    for (auto it = fi_bools.begin(), end = fi_bools.end(); it != end; ++it) {
      const face& face = faces[it->first];
      if (face[0] == he.first && face[1] == he.second) return it;
      if (face[1] == he.first && face[2] == he.second) return it;
      if (face[2] == he.first && face[0] == he.second) return it;
    }
    return fi_bools.end();
  }

  bool segment_crosses_the_plane(vertex_index s1, vertex_index s2, const face& f) const {
    const auto e1 = vertices[f[1]] - vertices[f[0]];
    const auto e2 = vertices[f[2]] - vertices[f[0]];

    const auto n = e1.cross(e2);
    const auto sign1 = n.dot(vertices[s1] - vertices[f[0]]);
    const auto sign2 = n.dot(vertices[s2] - vertices[f[0]]);
    return sign1 * sign2 < 0.0;
  }

  bool line_triangle_intersects(vertex_index s1, vertex_index s2, const face& f) const {
    const auto e1 = vertices[f[1]] - vertices[f[0]];
    const auto e2 = vertices[f[2]] - vertices[f[0]];

    const auto dir = vertices[s2] - vertices[s1];
    const auto p = dir.cross(e2);
    // det = [e1, dir, e2] (scalar triple product of dir, e2 and e1)
    const auto inv_det = 1.0 / p.dot(e1);
    const auto s = vertices[s1] - vertices[f[0]];
    const auto u = inv_det * s.dot(p);
    if (u < 0.0 || u > 1.0) {
      return false;
    }
    const auto q = s.cross(e1);
    const auto v = inv_det * dir.dot(q);
    if (v < 0.0 || v > 1.0 || u + v > 1.0) {
      return false;
    }
    if (u + v > 1.0) {
      return false;
    }

    return true;
  }

public:
  typedef std::pair<vertex_index, vertex_index> edge;

  mesh_defects_finder(const std::vector<Eigen::Vector3d>& vertices, const std::vector<face>& faces)
    : vertices(vertices)
    , faces(faces) {
  }

  std::vector<edge> non_manifold_edges() const {
    typedef std::pair<vertex_index, vertex_index> Edge;

    std::multiset<Edge> edges;
    std::set<Edge> non_manif_edges;

    for (auto& face : faces) {
      auto edge = common::make_sorted_pair(face[0], face[1]);
      edges.insert(edge);
      if (edges.count(edge) > 2) non_manif_edges.insert(edge);

      edge = common::make_sorted_pair(face[1], face[2]);
      edges.insert(edge);
      if (edges.count(edge) > 2) non_manif_edges.insert(edge);

      edge = common::make_sorted_pair(face[2], face[0]);
      edges.insert(edge);
      if (edges.count(edge) > 2) non_manif_edges.insert(edge);
    }

    std::vector<Edge> ret(non_manif_edges.begin(), non_manif_edges.end());

    return ret;
  }

  std::vector<vertex_index> non_manifold_vertices() const {
    std::vector<face_index_bools> v_fi_bools(vertices.size());

    for (int fi = 0; fi < faces.size(); fi++) {
      auto& face = faces[fi];
      v_fi_bools[face[0]].push_back(face_index_bool(fi, false));
      v_fi_bools[face[1]].push_back(face_index_bool(fi, false));
      v_fi_bools[face[2]].push_back(face_index_bool(fi, false));
    }

    std::vector<vertex_index> non_manif_vertices;

    for (int vi = 0; vi < vertices.size(); vi++) {
      face_index_bools& fi_bools = v_fi_bools[vi];

      if (fi_bools.size() == 0) {
        // Unreferrenced vertex
        continue;
      }

      face_index_bool& fi_bool = fi_bools[0];
      fi_bool.second = true;

      halfedge he = vertex_outgoing_halfedge(fi_bool.first, vi);
      while (true) {
        auto opp = opposite_halfedge(he);
        auto it = halfedge_face(fi_bools, opp);
        if (it == fi_bools.end() || it->second) {
          break;
        }
        face_index_bool& adj_fi_bool = *it;
        adj_fi_bool.second = true;
        he = vertex_outgoing_halfedge(adj_fi_bool.first, vi);
      }

      he = vertex_incoming_halfedge(fi_bool.first, vi);
      while (true) {
        auto opp = opposite_halfedge(he);
        auto it = halfedge_face(fi_bools, opp);
        if (it == fi_bools.end() || it->second) {
          break;
        }
        face_index_bool& adj_fi_bools = *it;
        adj_fi_bools.second = true;
        he = vertex_incoming_halfedge(adj_fi_bools.first, vi);
      }

      // Check if all faces are marked
      bool is_manif = std::all_of(fi_bools.begin(), fi_bools.end(), [](const face_index_bool& fi_bool) {
        return fi_bool.second;
      });
      if (!is_manif) non_manif_vertices.push_back(vi);
    }

    return non_manif_vertices;
  }

  std::vector<face> intersecting_faces() const {
    std::multimap<vertex_index, face> vf_map;

    for (auto& face : faces) {
      vf_map.insert({ face[0], { face[0], face[1], face[2] }});
      vf_map.insert({ face[1], { face[1], face[2], face[0] }});
      vf_map.insert({ face[2], { face[2], face[0], face[1] }});
    }

    std::vector<face> intersect_faces;

    for (int vi = 0; vi < vertices.size(); vi++) {
      auto vf_range = vf_map.equal_range(vi);
      for (auto vf_it1 = vf_range.first; vf_it1 != vf_range.second; ++vf_it1) {
        auto& f1 = vf_it1->second;

        for (auto vf_it2 = vf_range.first; vf_it2 != vf_range.second; ++vf_it2) {
          auto& f2 = vf_it2->second;

          if (f1[1] == f2[1] || f1[1] == f2[2] || f1[2] == f2[1]) {
            // Skip self pair and edge-sharing pairs.
            continue;
          }

          // Check if two faces are intersecting.
          if (segment_crosses_the_plane(f1[1], f1[2], f2) &&
              segment_crosses_the_plane(f2[1], f2[2], f1) &&
              (line_triangle_intersects(f1[1], f1[2], f2) ||
               line_triangle_intersects(f2[1], f2[2], f1))) {
            intersect_faces.push_back(f1);
            intersect_faces.push_back(f2);
          }
        }
      }
    }

    return intersect_faces;
  }
};

} // namespace isosurface
} // namespace polatory
