#pragma once

#include <algorithm>
#include <cmath>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <scalfmm/tree/box.hpp>

namespace polatory::fmm {

inline int fmm_tree_height(index_t points_estimated) {
  return std::max(3, static_cast<int>(std::round(std::log(points_estimated) / std::log(8))));
}

template <class Model, class Box>
Box make_box(const Model& model, const geometry::bboxNd<Model::kDim>& bbox) {
  auto a_bbox = bbox.transform(model.rbf().anisotropy());

  auto width = 1.01 * a_bbox.size().maxCoeff();
  if (width == 0.0) {
    width = 1.0;
  }

  typename Box::position_type center;
  for (auto i = 0; i < Model::kDim; ++i) {
    center.at(i) = a_bbox.center()(i);
  }

  return {width, center};
}

}  // namespace polatory::fmm
