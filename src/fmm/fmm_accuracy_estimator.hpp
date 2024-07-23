#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <polatory/fmm/interpolator_configuration.hpp>
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

namespace polatory::fmm {

template <class Rbf, class Kernel>
class fmm_accuracy_estimator {
  static constexpr int kDim{Rbf::kDim};
  using Bbox = geometry::bboxNd<kDim>;
  using Point = geometry::pointNd<kDim>;
  using Points = geometry::pointsNd<kDim>;
  using Vector = geometry::vectorNd<kDim>;

  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using SourceParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,
      /* outputs */ double, kn,  // should be 0
      /* variables */ index_t>;

  using TargetParticle = scalfmm::container::particle<
      /* position */ double, kDim,
      /* inputs */ double, km,  // should be 0
      /* outputs */ double, kn,
      /* variables */ index_t>;

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

  static constexpr int kClassic = interpolator_configuration::kClassic;
  static constexpr int kMaxOrder = 20;
  static constexpr index_t kMaxTargetSize = 10000;

 public:
  static interpolator_configuration find_best_configuration(const Rbf& rbf, double accuracy,
                                                            const SourceContainer& src_particles,
                                                            const Box& box, int tree_height) {
    const auto& config = rbf.interpolator_configuration();

    if (accuracy == std::numeric_limits<double>::infinity()) {
      return config;
    }

    // Errors at the data points are larger than those at randomly distributed points.

    auto src_size = static_cast<index_t>(src_particles.size());
    auto trg_size = std::min(src_size, kMaxTargetSize);
    TargetContainer trg_particles(trg_size);

    std::mt19937 gen;
    std::vector<index_t> src_indices(src_size);
    std::iota(std::begin(src_indices), std::end(src_indices), 0);
    std::shuffle(std::begin(src_indices), std::end(src_indices), gen);

    for (index_t idx = 0; idx < trg_size; idx++) {
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
    auto use_d = config.d != kClassic;
    auto best_d = config.d;
    auto last_error = std::numeric_limits<double>::infinity();
    for (auto order = config.order; order <= kMaxOrder; order++) {
      auto min_d = kClassic;
      auto max_d = kClassic;
      if (use_d) {
        // d = order - 1 give the same result as d = order - 2.
        min_d = best_d != kClassic ? std::max(best_d - 1, 3) : 3;
        max_d = best_d != kClassic ? std::min(best_d + 1, order - 2) : order - 2;
      }
      auto best_error = std::numeric_limits<double>::infinity();
      for (auto d = min_d; d <= max_d; d++) {
        auto approx = evaluate(rbf, src_particles, trg_particles, box, tree_height, order, d);
        auto error = numeric::absolute_error<Eigen::Infinity>(approx, exact);
        if (error <= accuracy) {
          return {order, d};
        }
        if (use_d) {
          if (error < best_error) {
            best_d = d;
            best_error = error;
          }
        } else {
          if (error > last_error) {
            use_d = true;
            order--;
          }
          last_error = error;
        }
      }
    }

    throw std::runtime_error("failed to construct an evaluator that meets the given accuracy");
  }

  static vectord evaluate(const Rbf& rbf, const SourceContainer& src_particles,
                          TargetContainer& trg_particles, const Box& box, int tree_height = 0,
                          int order = 0, int d = kClassic) {
    using namespace scalfmm::algorithms;

    auto trg_size = static_cast<index_t>(trg_particles.size());

    vectord potentials = vectord::Zero(kn * trg_size);
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

      for (index_t idx = 0; idx < trg_size; idx++) {
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
