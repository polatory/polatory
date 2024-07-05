#pragma once

#include <algorithm>
#include <limits>
#include <polatory/numeric/error.hpp>
#include <polatory/types.hpp>
#include <random>
#include <scalfmm/algorithms/fmm.hpp>
#include <scalfmm/algorithms/full_direct.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/interpolation/interpolation.hpp>
#include <scalfmm/operators/fmm_operators.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/cell.hpp>
#include <scalfmm/tree/for_each.hpp>
#include <scalfmm/tree/group_tree_view.hpp>
#include <scalfmm/tree/leaf_view.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace polatory::fmm {

template <class Rbf, class Kernel>
class fmm_accuracy_estimator {
  static constexpr int kDim{Rbf::kDim};
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

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

  using NearField = scalfmm::operators::near_field_operator<Kernel>;
  using Interpolator =
      scalfmm::interpolation::interpolator<double, kDim, Kernel, scalfmm::options::uniform_<>>;
  using FarField = scalfmm::operators::far_field_operator<Interpolator>;
  using FmmOperator = scalfmm::operators::fmm_operators<NearField, FarField>;
  using Position = typename SourceParticle::position_type;
  using Box = scalfmm::component::box<Position>;
  using Cell = scalfmm::component::cell<typename Interpolator::storage_type>;
  using SourceLeaf = scalfmm::component::leaf_view<SourceParticle>;
  using TargetLeaf = scalfmm::component::leaf_view<TargetParticle>;
  using SourceTree = scalfmm::component::group_tree_view<Cell, SourceLeaf, Box>;
  using TargetTree = scalfmm::component::group_tree_view<Cell, TargetLeaf, Box>;

  static constexpr int kMinimumOrder = 8;
  static constexpr int kMaximumOrder = 14;
  static constexpr index_t kNumTargetPoints = 4096;

 public:
  static int find_best_order(const Rbf& rbf, const std::vector<SourceParticle>& src_particles,
                             const Box& box, int tree_height, double accuracy) {
    if (accuracy == std::numeric_limits<double>::infinity()) {
      return kMinimumOrder;
    }

    std::vector<TargetParticle> trg_particles(kNumTargetPoints);

    auto center = box.center();
    auto radius = box.width(0) / 2.0;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist{-radius, radius};

    for (index_t idx = 0; idx < kNumTargetPoints; idx++) {
      auto& p = trg_particles.at(idx);
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = center.at(i) + dist(gen);
      }
      p.variables(idx);
    }

    auto exact = evaluate(rbf, src_particles, trg_particles, box, 0, 0);
    auto last_error = std::numeric_limits<double>::infinity();
    for (auto order = kMinimumOrder; order <= kMaximumOrder; order++) {
      auto approx = evaluate(rbf, src_particles, trg_particles, box, tree_height, order);
      auto error = numeric::absolute_error(approx, exact);
      if (error <= accuracy) {
        return order;
      }
      if (error > last_error) {
        break;
      }
      last_error = error;
    }

    throw std::runtime_error("failed to construct an evaluator that meets the given accuracy");
  }

  static vectord evaluate(const Rbf& rbf, const std::vector<SourceParticle>& src_particles,
                          std::vector<TargetParticle>& trg_particles, const Box& box,
                          int tree_height, int order) {
    using namespace scalfmm::algorithms;

    for (auto& p : trg_particles) {
      for (auto i = 0; i < kn; i++) {
        p.outputs(i) = 0.0;
      }
    }

    vectord potentials = vectord::Zero(kn * kNumTargetPoints);

    Kernel kernel(rbf);
    if (tree_height > 0) {
      NearField near_field(kernel, false);
      Interpolator interpolator(kernel, order, tree_height, box.width(0));
      FarField far_field(interpolator);
      FmmOperator fmm_operator(near_field, far_field);

      SourceTree src_tree(tree_height, order, box, 10, 10, src_particles);
      TargetTree trg_tree(tree_height, order, box, 10, 10, trg_particles);

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
      scalfmm::algorithms::full_direct(src_particles, trg_particles, kernel);

      for (index_t idx = 0; idx < kNumTargetPoints; idx++) {
        const auto& p = trg_particles.at(idx);
        for (auto i = 0; i < kn; i++) {
          potentials(kn * idx + i) = p.outputs(i);
        }
      }
    }

    return potentials;
  }
};

}  // namespace polatory::fmm
