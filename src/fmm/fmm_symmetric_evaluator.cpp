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

#include "kernel.hpp"
#include "utility.hpp"

namespace polatory::fmm {

template <int Order>
class fmm_symmetric_evaluator<Order>::impl {
  using Particle = scalfmm::container::particle<
      /* position */ double, 3,
      /* inputs */ double, 1,
      /* outputs */ double, 1,
      /* variables */ index_t>;

  using Kernel = kernel;
  using NearField = scalfmm::operators::near_field_operator<Kernel>;
  using Interpolator =
      scalfmm::interpolation::interpolator<double, 3, Kernel, scalfmm::options::chebyshev_<>>;
  using FarField = scalfmm::operators::far_field_operator<Interpolator>;
  using FmmOperator = scalfmm::operators::fmm_operators<NearField, FarField>;
  using Position = typename Particle::position_type;
  using Box = scalfmm::component::box<Position>;
  using Cell = scalfmm::component::cell<typename Interpolator::storage_type>;
  using Leaf = scalfmm::component::leaf_view<Particle>;
  using Tree = scalfmm::component::group_tree_view<Cell, Leaf, Box>;

 public:
  impl(const model& model, const geometry::bbox3d& bbox)
      : model_(model),
        kernel_(model.rbf()),
        order_(Order),
        box_(make_box<Box>(model, bbox)),
        near_field_(kernel_) {}

  common::valuesd evaluate() const {
    using namespace scalfmm::algorithms;

    if (prepare()) {
      tree_->reset_multipoles();
      tree_->reset_locals();
      tree_->reset_outputs();
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](  //
          *tree_, *fmm_operator_, p2m | m2m | m2l | l2l | l2p | p2p);
    }

    return potentials();
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = points.rows();

    particles_.resize(n_points_);

    auto a = model_.rbf().anisotropy();
    for (index_t idx = 0; idx < n_points_; idx++) {
      auto& p = particles_.at(idx);
      auto ap = geometry::transform_point(a, points.row(idx));
      p.position() = Position{ap(0), ap(1), ap(2)};
      p.variables(idx);
    }

    tree_.reset(nullptr);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    POLATORY_ASSERT(weights.rows() == n_points_);

    if (!tree_) {
      for (index_t idx = 0; idx < n_points_; idx++) {
        auto& p = particles_.at(idx);
        p.inputs(0) = weights(idx);
      }
    } else {
      scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename Leaf::proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            p.inputs(0) = weights(idx);
                                          }
                                        });
    }
  }

 private:
  common::valuesd potentials() const {
    common::valuesd potentials = common::valuesd::Zero(n_points_);

    if (tree_) {
      auto a = model_.rbf().evaluate(geometry::vector3d::Zero());
      scalfmm::component::for_each_leaf(std::cbegin(*tree_), std::cend(*tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename Leaf::const_proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            potentials(idx) = p.outputs(0) + p.inputs(0) * a;
                                          }
                                        });
    }

    return potentials;
  }

  bool prepare() const {
    if (n_points_ == 0) {
      interpolator_.reset(nullptr);
      far_field_.reset(nullptr);
      fmm_operator_.reset(nullptr);
      tree_.reset(nullptr);
      return false;
    }

    auto tree_height = fmm_tree_height(n_points_);

    auto tree_height_changed = tree_height_ != tree_height;
    tree_height_ = tree_height;

    if (!interpolator_ || tree_height_changed) {
      interpolator_ = std::make_unique<Interpolator>(kernel_, order_, tree_height, box_.width(0));
      far_field_ = std::make_unique<FarField>(*interpolator_);
      fmm_operator_ = std::make_unique<FmmOperator>(near_field_, *far_field_);
    }

    if (!tree_) {
      tree_ = std::make_unique<Tree>(tree_height, order_, box_, 10, 10, particles_);
      particles_.clear();
      particles_.shrink_to_fit();
    } else if (tree_height_changed) {
      std::vector<Particle> particles(n_points_);

      scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_), [&](auto& leaf) {
        for (auto p_ref : leaf) {
          auto p = typename Leaf::proxy_type(p_ref);
          auto idx = std::get<0>(p.variables());
          auto& new_p = particles.at(idx);
          for (auto i = 0; i < 3; i++) {
            new_p.position(i) = p.position(i);
          }
          new_p.inputs(0) = p.inputs(0);
          new_p.variables(idx);
        }
      });

      tree_ = std::make_unique<Tree>(tree_height, order_, box_, 10, 10, particles);
    }

    return true;
  }

  const model& model_;
  const Kernel kernel_;
  const int order_;
  const Box box_;
  const NearField near_field_;

  index_t n_points_{};
  mutable std::vector<Particle> particles_;
  mutable int tree_height_{};
  mutable std::unique_ptr<Interpolator> interpolator_;
  mutable std::unique_ptr<FarField> far_field_;
  mutable std::unique_ptr<FmmOperator> fmm_operator_;
  mutable std::unique_ptr<Tree> tree_;
};

template <int Order>
fmm_symmetric_evaluator<Order>::fmm_symmetric_evaluator(const model& model,
                                                        const geometry::bbox3d& bbox)
    : pimpl_(std::make_unique<impl>(model, bbox)) {}

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
