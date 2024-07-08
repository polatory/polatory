#pragma once

#include <algorithm>
#include <limits>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/numeric/error.hpp>
#include <polatory/types.hpp>
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
  static constexpr index_t kTargetSize = 4096;

 public:
  static int find_best_order(const Rbf& rbf, const Bbox& bbox, double accuracy,
                             const SourceContainer& src_particles, const Box& box,
                             int tree_height) {
    if (accuracy == std::numeric_limits<double>::infinity()) {
      return kMinimumOrder;
    }

    TargetContainer trg_particles(kTargetSize);

    auto a = rbf.anisotropy();
    Point center = bbox.center();
    Vector radius = bbox.width() / 2.0;
    for (index_t idx = 0; idx < kTargetSize; idx++) {
      auto p = trg_particles.at(idx);
      auto ap = geometry::transform_point<kDim>(a, center + radius.cwiseProduct(Point::Random()));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
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

  static vectord evaluate(const Rbf& rbf, const SourceContainer& src_particles,
                          TargetContainer& trg_particles, const Box& box, int tree_height,
                          int order) {
    using namespace scalfmm::algorithms;

    vectord potentials = vectord::Zero(kn * kTargetSize);
    trg_particles.reset_outputs();

    Kernel kernel(rbf);
    if (tree_height > 0) {
      NearField near_field(kernel, false);
      Interpolator interpolator(kernel, order, tree_height, box.width(0));
      FarField far_field(interpolator);
      FmmOperator fmm_operator(near_field, far_field);

      SourceTree src_tree(tree_height, order, box, 10, 10, src_particles, true);
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

      for (index_t idx = 0; idx < kTargetSize; idx++) {
        const auto p = trg_particles.at(idx);
        for (auto i = 0; i < kn; i++) {
          potentials(kn * idx + i) = p.outputs(i);
        }
      }
    }

    return potentials;
  }
};

}  // namespace polatory::fmm
