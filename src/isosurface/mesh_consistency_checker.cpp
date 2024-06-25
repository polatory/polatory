#pragma once

#include <igl/AABB.h>
#include <igl/facet_components.h>
#include <polatory/isosurface/mesh_consistency_checker.h>

#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace polatory::isosurface {

mesh_consistency_checker::mesh_consistency_checker(const vertices_type& vertices,
                                                   const faces_type& faces)
    : vertices_(vertices), faces_(faces) {
  Eigen::VectorXi component_ids;
  auto num_components = igl::facet_components(faces, component_ids);

  std::vector<index_t> representative_face(num_components);
  for (index_t fi = 0; fi < faces.rows(); ++fi) {
    auto cid = component_ids(fi);
    representative_face.at(cid) = fi;
  }

  std::vector<bool> deleted(num_components, false);

  igl::AABB<vertices_type, 3> tree;
  tree.init(vertices, faces);
  std::vector<igl::Hit> hits;
  for (auto src_cid = 0; src_cid < num_components - 1; ++src_cid) {
    if (deleted.at(src_cid)) {
      continue;
    }

    auto src_face = representative_face.at(src_cid);
    geometry::point3d p = face_centroid(src_face);
    for (auto trg_cid = src_cid + 1; trg_cid < num_components; ++trg_cid) {
      if (deleted.at(src_cid)) {
        break;
      }
      if (deleted.at(trg_cid)) {
        continue;
      }

      auto trg_face = representative_face.at(trg_cid);
      geometry::point3d q = face_centroid(trg_face);
      tree.intersect_ray(vertices_, faces_, p, q - p, hits);
      std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) { return a.t < b.t; });
      for (std::size_t left = 0; left < hits.size() - 1; ++left) {
        auto right = left + 1;
        auto left_face = hits.at(left).id;
        auto right_face = hits.at(right).id;
        auto left_cid = component_ids(left_face);
        auto right_cid = component_ids(right_face);
        if (left_cid == right_cid) {
          continue;
        }
        if (face_normal(left_face).dot(face_normal(right_face)) >= 0.0) {
          continue;
        }

        auto new_cid = std::min(left_cid, right_cid);
        auto old_cid = std::max(left_cid, right_cid);
        for (auto& cid : component_ids) {
          if (cid == old_cid) {
            cid = new_cid;
          }
        }
        deleted.at(old_cid) = true;
      }
    }
  }
};

}  // namespace polatory::isosurface
