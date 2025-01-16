#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/types.hpp>
#include <random>
#include <scalfmm/algorithms/fmm.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/interpolation/interpolation.hpp>
#include <scalfmm/operators/fmm_operators.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/cell.hpp>
#include <scalfmm/tree/for_each.hpp>
#include <scalfmm/tree/group_tree_view.hpp>
#include <scalfmm/tree/leaf_view.hpp>
#include <scalfmm/utils/sort.hpp>
#include <stdexcept>
#include <tuple>

#include "full_direct.hpp"
#include "interpolator_configuration.hpp"

namespace polatory::fmm {

template <class Rbf, class Kernel>
class FmmAccuracyEstimator {
  static constexpr int kDim{Rbf::kDim};
  using Bbox = geometry::Bbox<kDim>;
  using Point = geometry::Point<kDim>;
  using Points = geometry::Points<kDim>;
  using Vector = geometry::Vector<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using SourceParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,  // should be 0
      /* variables */ Index>;

  using TargetParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,  // should be 0
      /* outputs */ double, kn,
      /* variables */ Index>;

  using SourceContainer = scalfmm::container::particle_container<SourceParticle>;
  using TargetContainer = scalfmm::container::particle_container<TargetParticle>;

  using NearField = scalfmm::operators::near_field_operator<Kernel>;
  using Interpolator = scalfmm::interpolation::interpolator<double, kDim, Kernel,
                                                            scalfmm::options::modified_uniform_>;
  using FarField = scalfmm::operators::far_field_operator<Interpolator>;
  using FmmOperator = scalfmm::operators::fmm_operators<NearField, FarField>;
  using Position = typename SourceParticle::position_type;
  using Box = scalfmm::component::box<Position>;
  using Cell = scalfmm::component::cell<typename Interpolator::storage_type>;
  using SourceLeaf = scalfmm::component::leaf_view<SourceParticle>;
  using TargetLeaf = scalfmm::component::leaf_view<TargetParticle>;
  using SourceTree = scalfmm::component::group_tree_view<Cell, SourceLeaf, Box>;
  using TargetTree = scalfmm::component::group_tree_view<Cell, TargetLeaf, Box>;

  static constexpr int kClassic = InterpolatorConfiguration::kClassic;
  static constexpr Index kMaxTargetSize = 10000;

 public:
  static InterpolatorConfiguration find_best_configuration(const Rbf& rbf, double accuracy,
                                                           const SourceContainer& src_particles,
                                                           const Box& box, int tree_height) {
    if (accuracy == std::numeric_limits<double>::infinity()) {
      return {.tree_height = tree_height, .order = 6, .d = kClassic};
    }
    if (accuracy == 0.0) {
      return {.tree_height = tree_height, .order = 12, .d = 8};
    }

    // Errors at the data points are larger than those at randomly distributed points.

    auto src_size = static_cast<Index>(src_particles.size());
    auto trg_size = std::min(src_size, kMaxTargetSize);
    TargetContainer trg_particles(trg_size);

    std::mt19937 gen;
    std::vector<Index> src_indices(src_size);
    std::iota(std::begin(src_indices), std::end(src_indices), 0);
    std::shuffle(std::begin(src_indices), std::end(src_indices), gen);

    for (Index idx = 0; idx < trg_size; idx++) {
      auto p = trg_particles.at(idx);
      auto src_idx = src_indices.at(idx);
      const auto src_p = src_particles.at(src_idx);
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = src_p.position(i);
      }
      p.variables(idx);
    }

    scalfmm::utils::sort_container(box, tree_height - 1, trg_particles);

    auto exact = evaluate(rbf, src_particles, trg_particles, box);
    for (auto order = 8; order <= 20; order += 2) {
      auto min_d = order >= 12 ? 7 : kClassic;
      auto max_d = order >= 12 ? 9 : kClassic;
      for (auto d = min_d; d <= max_d; d++) {
        auto approx = evaluate(rbf, src_particles, trg_particles, box, tree_height, order, d);
        auto error = numeric::absolute_error<Eigen::Infinity>(approx, exact);
        if (error <= accuracy) {
          return {.tree_height = tree_height, .order = order, .d = d};
        }
      }
    }

    throw std::runtime_error("failed to construct an evaluator that meets the desired accuracy");
  }

  static VecX evaluate(const Rbf& rbf, const SourceContainer& src_particles,
                       TargetContainer& trg_particles, const Box& box, int tree_height = 0,
                       int order = 0, int d = kClassic) {
    using namespace scalfmm::algorithms;

    auto trg_size = static_cast<Index>(trg_particles.size());

    VecX potentials = VecX::Zero(kn * trg_size);
    trg_particles.reset_outputs();

    Kernel kernel(rbf);
    if (tree_height > 0) {
      NearField near_field(kernel, false);
      Interpolator interpolator(kernel, order, tree_height, box.width(0), d);
      FarField far_field(interpolator);
      FmmOperator fmm_operator(near_field, far_field);

      SourceTree src_tree(tree_height, order, box, 10, 10, src_particles, true);
      TargetTree trg_tree(tree_height, order, box, 10, 10, trg_particles, true);

      scalfmm::list::omp::build_interaction_lists(src_tree, trg_tree, 1, false);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)]  //
          (src_tree, trg_tree, fmm_operator, p2m | m2m | m2l | l2l | l2p | p2p);

      scalfmm::component::for_each_leaf(std::cbegin(trg_tree), std::cend(trg_tree),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename TargetLeaf::const_proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            for (auto i = 0; i < kn; i++) {
                                              potentials(kn * idx + i) = p.outputs(i);
                                            }
                                          }
                                        });
    } else {
      full_direct(src_particles, trg_particles, kernel);

      for (Index idx = 0; idx < trg_size; idx++) {
        const auto p = trg_particles.at(idx);
        auto orig_idx = std::get<0>(p.variables());
        for (auto i = 0; i < kn; i++) {
          potentials(kn * orig_idx + i) = p.outputs(i);
        }
      }
    }

    return potentials;
  }
};

}  // namespace polatory::fmm
