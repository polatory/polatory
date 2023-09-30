#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/rbf/biharmonic2d.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/multiquadric1.hpp>
#include <polatory/rbf/reference/cov_gaussian.hpp>
#include <polatory/rbf/reference/cov_spherical.hpp>
#include <polatory/rbf/reference/triharmonic3d.hpp>
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

template <class Model, class Kernel>
class fmm_generic_symmetric_evaluator<Model, Kernel>::impl {
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
  impl(const Model& model, const geometry::bbox3d& bbox, precision prec)
      : model_(model),
        kernel_(model.rbf()),
        order_(static_cast<int>(prec)),
        box_(make_box<Model, Box>(model, bbox)),
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
          *tree_, *fmm_operator_, m2l | p2p);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](  //
          *tree_, *fmm_operator_, l2l | l2p);
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

  const Model& model_;
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

template <class Model, class Kernel>
fmm_generic_symmetric_evaluator<Model, Kernel>::fmm_generic_symmetric_evaluator(
    const Model& model, const geometry::bbox3d& bbox, precision prec)
    : impl_(std::make_unique<impl>(model, bbox, prec)) {}

template <class Model, class Kernel>
fmm_generic_symmetric_evaluator<Model, Kernel>::~fmm_generic_symmetric_evaluator() = default;

template <class Model, class Kernel>
common::valuesd fmm_generic_symmetric_evaluator<Model, Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Model, class Kernel>
void fmm_generic_symmetric_evaluator<Model, Kernel>::set_points(const geometry::points3d& points) {
  impl_->set_points(points);
}

template <class Model, class Kernel>
void fmm_generic_symmetric_evaluator<Model, Kernel>::set_weights(
    const Eigen::Ref<const common::valuesd>& weights) {
  impl_->set_weights(weights);
}

#define IMPLEMENT2(MODEL, DIM)                                                                  \
  template class fmm_generic_symmetric_evaluator<MODEL, kernel<typename MODEL::rbf_type, DIM>>; \
  template class fmm_generic_symmetric_evaluator<MODEL,                                         \
                                                 hessian_kernel<typename MODEL::rbf_type, DIM>>;

#define IMPLEMENT(MODEL) \
  IMPLEMENT2(MODEL, 1);  \
  IMPLEMENT2(MODEL, 2);  \
  IMPLEMENT2(MODEL, 3);

IMPLEMENT(model<rbf::biharmonic2d>);
IMPLEMENT(model<rbf::biharmonic3d>);
IMPLEMENT(model<rbf::cov_exponential>);
IMPLEMENT(model<rbf::cov_spheroidal3>);
IMPLEMENT(model<rbf::cov_spheroidal5>);
IMPLEMENT(model<rbf::cov_spheroidal7>);
IMPLEMENT(model<rbf::cov_spheroidal9>);
IMPLEMENT(model<rbf::multiquadric1>);
IMPLEMENT(model<rbf::reference::cov_gaussian>);
IMPLEMENT(model<rbf::reference::cov_spherical>);
IMPLEMENT(model<rbf::reference::triharmonic3d>);

}  // namespace polatory::fmm
