#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <scalfmm/algorithms/fmm.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/interpolation/interpolation.hpp>
#include <scalfmm/operators/fmm_operators.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/cell.hpp>
#include <scalfmm/tree/for_each.hpp>
#include <scalfmm/tree/group_tree_view.hpp>
#include <scalfmm/tree/leaf_view.hpp>
#include <tuple>
#include <vector>

#include "fmm_rbf_kernel.hpp"

namespace polatory::fmm {

template <int Order>
class fmm_symmetric_evaluator<Order>::impl {
  using Particle = scalfmm::container::particle<
      /* position */ double, 3,
      /* inputs */ double, 1,
      /* outputs */ double, 1,
      /* variables */ index_t>;

  using NearField = scalfmm::operators::near_field_operator<fmm_rbf_kernel>;
  using Interpolator = scalfmm::interpolation::interpolator<
      double, 3, fmm_rbf_kernel, scalfmm::options::chebyshev_<scalfmm::options::low_rank_>>;
  using FarField = scalfmm::operators::far_field_operator<Interpolator>;
  using FmmOperator = scalfmm::operators::fmm_operators<NearField, FarField>;
  using Position = typename Particle::position_type;
  using Box = scalfmm::component::box<Position>;
  using Cell = scalfmm::component::cell<typename Interpolator::storage_type>;
  using Leaf = scalfmm::component::leaf_view<Particle>;
  using Tree = scalfmm::component::group_tree_view<Cell, Leaf, Box>;

 private:
  Box make_box(const model& model, const geometry::bbox3d& bbox) {
    auto a_bbox = bbox.transform(model.rbf().anisotropy());
    auto width = 1.01 * a_bbox.size().maxCoeff();
    if (width == 0.0) {
      width = 1.0;
    }
    auto center = a_bbox.center();
    return {width, {center(0), center(1), center(2)}};
  }

 public:
  impl(const model& model, int tree_height, const geometry::bbox3d& bbox)
      : model_(model),
        rbf_kernel_(model.rbf()),
        order_(Order),
        tree_height_(tree_height),
        box_(make_box(model, bbox)),
        near_field_(rbf_kernel_),
        interpolator_(rbf_kernel_, order_, tree_height, box_.width(0)),
        far_field_(interpolator_),
        fmm_operator_(near_field_, far_field_) {}

  common::valuesd evaluate() const {
    using namespace scalfmm::algorithms;

    tree_->reset_multipoles();
    tree_->reset_locals();
    tree_->reset_outputs();

    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](
        *tree_, fmm_operator_, p2m | m2m | m2l | l2l | l2p | p2p);

    auto self_potential = model_.rbf().evaluate_isotropic(geometry::vector3d::Zero());
    return potentials() + weights_ * self_potential;
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = points.rows();

    auto a = model_.rbf().anisotropy();

    std::vector<Particle> particles(n_points_);
    for (index_t i = 0; i < n_points_; i++) {
      auto& p = particles.at(i);
      auto ap = geometry::transform_point(a, points.row(i));
      p.position() = Position{ap(0), ap(1), ap(2)};
      p.inputs().at(0) = 0.0;
      p.outputs().at(0) = 0.0;
      p.variables(i);
    }

    tree_ = std::make_unique<Tree>(tree_height_, order_, box_, 10, 10, particles);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    POLATORY_ASSERT(weights.rows() == n_points_);

    scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_), [&](const auto& leaf) {
      // loop on the particles of the leaf
      for (auto p_ref : leaf) {
        // build a particle
        auto p = typename Leaf::proxy_type(p_ref);
        auto idx = std::get<0>(p.variables());
        p.inputs().at(0).get() = weights(idx);
      }
    });

    weights_ = weights;
  }

 private:
  common::valuesd potentials() const {
    common::valuesd potentials(n_points_);

    scalfmm::component::for_each_leaf(std::cbegin(*tree_), std::cend(*tree_),
                                      [&](const auto& leaf) {
                                        // loop on the particles of the leaf
                                        for (auto p_ref : leaf) {
                                          // build a particle
                                          auto p = typename Leaf::const_proxy_type(p_ref);
                                          auto idx = std::get<0>(p.variables());
                                          potentials(idx) = p.outputs().at(0);
                                        }
                                      });

    return potentials;
  }

  const model& model_;
  const fmm_rbf_kernel rbf_kernel_;
  const int order_;
  const int tree_height_;

  index_t n_points_{};
  common::valuesd weights_;

  mutable Box box_;
  mutable NearField near_field_;
  mutable Interpolator interpolator_;
  mutable FarField far_field_;
  mutable FmmOperator fmm_operator_;
  mutable std::unique_ptr<Tree> tree_;
};

template <int Order>
fmm_symmetric_evaluator<Order>::fmm_symmetric_evaluator(const model& model, int tree_height,
                                                        const geometry::bbox3d& bbox)
    : pimpl_(std::make_unique<impl>(model, tree_height, bbox)) {}

template <int Order>
fmm_symmetric_evaluator<Order>::~fmm_symmetric_evaluator() = default;

template <int Order>
common::valuesd fmm_symmetric_evaluator<Order>::evaluate() const {
  return pimpl_->evaluate();
}

template <int Order>
void fmm_symmetric_evaluator<Order>::set_points(const geometry::points3d& points) {
  pimpl_->set_points(points);
}

template <int Order>
void fmm_symmetric_evaluator<Order>::set_weights(const Eigen::Ref<const common::valuesd>& weights) {
  pimpl_->set_weights(weights);
}

template class fmm_symmetric_evaluator<6>;
template class fmm_symmetric_evaluator<10>;

}  // namespace polatory::fmm
