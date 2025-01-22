#pragma once

#include <algorithm>
#include <cmath>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/types.hpp>
#include <scalfmm/tree/box.hpp>

namespace polatory::fmm {

template <int Dim>
int fmm_tree_height(Index n_points) {
  return std::max(2,
                  static_cast<int>(std::round(std::log(n_points) / std::log(std::pow(2.0, Dim)))));
}

// IMPORTANT: Keep in sync with the function in include/polatory/fmm/resource.hpp.
template <class Rbf, class Box>
Box make_box(const Rbf& rbf, const geometry::Bbox<Rbf::kDim>& bbox) {
  auto a_bbox = bbox.transform(rbf.anisotropy());

  auto width = 1.01 * a_bbox.width().maxCoeff();
  if (width == 0.0) {
    width = 1.0;
  }

  typename Box::position_type center;
  for (auto i = 0; i < Rbf::kDim; ++i) {
    center.at(i) = a_bbox.center()(i);
  }

  return {width, center};
}

}  // namespace polatory::fmm
