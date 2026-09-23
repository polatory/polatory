#pragma once

#include <polatory/geometry/bbox3d.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/utils/sort.hpp>

#include "kernel_cost.hpp"

namespace polatory::fmm {

// Cost of one frequency component of the FFT-based M2L, i.e. km * kn complex products, relative
// to one near-field pair evaluation of the given kernel. Relative to Kernel<Biharmonic3D<Dim>>,
// one frequency component alone was measured between 0.9 and 3.6 across machines and orders;
// the larger value also accounts for the passes the model omits (P2M, M2M, L2L, L2P and the tree
// construction).
template <class Kernel>
inline constexpr double kM2LProductCostInPairs =
    static_cast<double>(Kernel::km * Kernel::kn) * 2.5 / kKernelCost<Kernel>;

// Sorts the particles by their Morton index at the deepest level at which neither the 64-bit index
// nor the shift 1 << level in scalfmm::index::get_morton_index overflows, so that they are
// sorted for every tree height.
template <class Box, class Container>
void sort_particles(const Box& box, Container& particles) {
  scalfmm::utils::sort_container(box, 63 / Box::dimension, particles);
}

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
