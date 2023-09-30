#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
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
#include <type_traits>
#include <vector>

#include "utility.hpp"

namespace polatory::fmm {

template <int Order, class Kernel>
class fmm_generic_symmetric_evaluator<Order, Kernel>::impl {
  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

  using Particle = scalfmm::container::particle<
      /* position */ double, 3,
      /* inputs */ double, km,
      /* outputs */ double, kn,
      /* variables */ index_t>;

  using NearField = scalfmm::operators::near_field_operator<Kernel>;
  using Interpolator =
      scalfmm::interpolation::interpolator<double, 3, Kernel, scalfmm::options::uniform_<>>;
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
      if (!tree_->is_interaction_m2l_lists_built()) {
        scalfmm::list::sequential::build_m2l_interaction_list(*tree_, *tree_, 1);
      }
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](  //
          *tree_, *fmm_operator_, p2m | m2m);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)](  //
          *tree_, *fmm_operator_, m2l);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](  //
          *tree_, *fmm_operator_, l2l | l2p);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)](  //
          *tree_, *fmm_operator_, p2p);
      handle_self_interaction();
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
    POLATORY_ASSERT(weights.rows() == km * n_points_);

    if (!tree_) {
      for (index_t idx = 0; idx < n_points_; idx++) {
        auto& p = particles_.at(idx);
        for (auto i = 0; i < km; i++) {
          p.inputs(i) = weights(km * idx + i);
        }
      }
    } else {
      scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename Leaf::proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            for (auto i = 0; i < km; i++) {
                                              p.inputs(i) = weights(km * idx + i);
                                            }
                                          }
                                        });
    }
  }

 private:
  void handle_self_interaction() const {
    scalfmm::container::point<double, 3> x{};
    auto k = kernel_.evaluate(x, x);

    scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_), [&](const auto& leaf) {
      for (auto p_ref : leaf) {
        auto p = typename Leaf::proxy_type(p_ref);
        for (auto i = 0; i < kn; i++) {
          for (auto j = 0; j < km; j++) {
            p.outputs(i) += p.inputs(j) * k.at(km * i + j);
          }
        }
      }
    });
  }

  common::valuesd potentials() const {
    common::valuesd potentials = common::valuesd::Zero(kn * n_points_);

    if (tree_) {
      scalfmm::component::for_each_leaf(std::cbegin(*tree_), std::cend(*tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename Leaf::const_proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            for (auto i = 0; i < kn; i++) {
                                              potentials(kn * idx + i) = p.outputs(i);
                                            }
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
      reset_tree();
      tree_height_ = 0;
      return false;
    }

    auto tree_height = fmm_tree_height(n_points_);
    if (tree_height_ != tree_height) {
      interpolator_ = std::make_unique<Interpolator>(kernel_, order_, tree_height, box_.width(0));
      far_field_ = std::make_unique<FarField>(*interpolator_);
      fmm_operator_ = std::make_unique<FmmOperator>(near_field_, *far_field_);
      reset_tree();
      tree_height_ = tree_height;
    }

    if (!tree_) {
      tree_ = std::make_unique<Tree>(tree_height, order_, box_, 10, 10, particles_);
      particles_.clear();
      particles_.shrink_to_fit();
    }

    return true;
  }

  void reset_tree() const {
    if (!tree_) {
      return;
    }

    particles_.resize(n_points_);

    scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_), [&](const auto& leaf) {
      for (auto p_ref : leaf) {
        auto p = typename Leaf::proxy_type(p_ref);
        auto idx = std::get<0>(p.variables());
        auto& new_p = particles_.at(idx);
        for (auto i = 0; i < 3; i++) {
          new_p.position(i) = p.position(i);
        }
        for (auto i = 0; i < km; i++) {
          new_p.inputs(i) = p.inputs(i);
        }
        new_p.variables(idx);
      }
    });

    tree_.reset(nullptr);
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

template <int Order, class Kernel>
fmm_generic_symmetric_evaluator<Order, Kernel>::fmm_generic_symmetric_evaluator(
    const model& model, const geometry::bbox3d& bbox)
    : pimpl_(std::make_unique<impl>(model, bbox)) {}

template <int Order, class Kernel>
fmm_generic_symmetric_evaluator<Order, Kernel>::~fmm_generic_symmetric_evaluator() = default;

template <int Order, class Kernel>
common::valuesd fmm_generic_symmetric_evaluator<Order, Kernel>::evaluate() const {
  return pimpl_->evaluate();
}

template <int Order, class Kernel>
void fmm_generic_symmetric_evaluator<Order, Kernel>::set_points(const geometry::points3d& points) {
  pimpl_->set_points(points);
}

template <int Order, class Kernel>
void fmm_generic_symmetric_evaluator<Order, Kernel>::set_weights(
    const Eigen::Ref<const common::valuesd>& weights) {
  pimpl_->set_weights(weights);
}

#define INSTANTIATE(KERNEL)                                      \
  template class fmm_generic_symmetric_evaluator<6, KERNEL<1>>;  \
  template class fmm_generic_symmetric_evaluator<10, KERNEL<1>>; \
  template class fmm_generic_symmetric_evaluator<6, KERNEL<2>>;  \
  template class fmm_generic_symmetric_evaluator<10, KERNEL<2>>; \
  template class fmm_generic_symmetric_evaluator<6, KERNEL<3>>;  \
  template class fmm_generic_symmetric_evaluator<10, KERNEL<3>>;

INSTANTIATE(kernel)
INSTANTIATE(hessian_kernel)

}  // namespace polatory::fmm
