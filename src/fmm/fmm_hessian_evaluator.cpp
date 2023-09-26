#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_hessian_evaluator.hpp>
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

#include "hessian_kernel.hpp"
#include "utility.hpp"

namespace polatory::fmm {

template <int Order, int Dim>
class fmm_hessian_evaluator<Order, Dim>::impl {
  using SourceParticle = scalfmm::container::particle<
      /* position */ double, 3,
      /* inputs */ double, Dim,
      /* outputs */ double, Dim,  // should be 0
      /* variables */ index_t>;

  using TargetParticle = scalfmm::container::particle<
      /* position */ double, 3,
      /* inputs */ double, Dim,  // should be 0
      /* outputs */ double, Dim,
      /* variables */ index_t>;

  using Kernel = hessian_kernel<Dim>;
  using NearField = scalfmm::operators::near_field_operator<Kernel>;
  using Interpolator =
      scalfmm::interpolation::interpolator<double, 3, Kernel, scalfmm::options::uniform_<>>;
  using FarField = scalfmm::operators::far_field_operator<Interpolator>;
  using FmmOperator = scalfmm::operators::fmm_operators<NearField, FarField>;
  using Position = typename SourceParticle::position_type;
  using Box = scalfmm::component::box<Position>;
  using Cell = scalfmm::component::cell<typename Interpolator::storage_type>;
  using SourceLeaf = scalfmm::component::leaf_view<SourceParticle>;
  using TargetLeaf = scalfmm::component::leaf_view<TargetParticle>;
  using SourceTree = scalfmm::component::group_tree_view<Cell, SourceLeaf, Box>;
  using TargetTree = scalfmm::component::group_tree_view<Cell, TargetLeaf, Box>;

 public:
  impl(const model& model, const geometry::bbox3d& bbox)
      : model_(model),
        kernel_(model.rbf()),
        order_(Order + 2),
        box_(make_box<Box>(model, bbox)),
        near_field_(kernel_, false) {}

  common::valuesd evaluate() const {
    using namespace scalfmm::algorithms;

    if (prepare()) {
      if (multipole_dirty_) {
        src_tree_->reset_multipoles();
        scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](
            *src_tree_, *fmm_operator_, p2m | m2m);
        multipole_dirty_ = false;
      }

      trg_tree_->reset_locals();
      trg_tree_->reset_outputs();
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](  //
          *src_tree_, *trg_tree_, *fmm_operator_, m2l | l2l | l2p | p2p);
    }

    return potentials();
  }

  void set_field_points(const geometry::points3d& points) {
    n_fld_points_ = points.rows();

    trg_particles_.resize(n_fld_points_);

    auto a = model_.rbf().anisotropy();
    for (index_t idx = 0; idx < n_fld_points_; idx++) {
      auto& p = trg_particles_.at(idx);
      auto ap = geometry::transform_point(a, points.row(idx));
      p.position() = Position{ap(0), ap(1), ap(2)};
      p.variables(idx);
    }

    trg_tree_.reset(nullptr);
  }

  void set_source_points(const geometry::points3d& points) {
    n_src_points_ = points.rows();

    src_particles_.resize(n_src_points_);

    auto a = model_.rbf().anisotropy();
    for (index_t idx = 0; idx < n_src_points_; idx++) {
      auto& p = src_particles_.at(idx);
      auto ap = geometry::transform_point(a, points.row(idx));
      p.position() = Position{ap(0), ap(1), ap(2)};
      p.variables(idx);
    }

    src_tree_.reset(nullptr);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    using namespace scalfmm::algorithms;

    POLATORY_ASSERT(weights.rows() == Dim * n_src_points_);

    if (!src_tree_) {
      for (index_t idx = 0; idx < n_src_points_; idx++) {
        auto& p = src_particles_.at(idx);
        for (auto i = 0; i < Dim; i++) {
          p.inputs(i) = weights(Dim * idx + i);
        }
      }
    } else {
      scalfmm::component::for_each_leaf(std::begin(*src_tree_), std::end(*src_tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename SourceLeaf::proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            for (auto i = 0; i < Dim; i++) {
                                              p.inputs(i) = weights(Dim * idx + i);
                                            }
                                          }
                                        });
      multipole_dirty_ = true;
    }
  }

 private:
  common::valuesd potentials() const {
    common::valuesd potentials(Dim * n_fld_points_);

    if (trg_tree_) {
      scalfmm::component::for_each_leaf(std::cbegin(*trg_tree_), std::cend(*trg_tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename TargetLeaf::const_proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            for (auto i = 0; i < Dim; i++) {
                                              potentials(Dim * idx + i) = p.outputs(i);
                                            }
                                          }
                                        });
    }

    return potentials;
  }

  bool prepare() const {
    if (n_src_points_ == 0 || n_fld_points_ == 0) {
      interpolator_.reset(nullptr);
      far_field_.reset(nullptr);
      fmm_operator_.reset(nullptr);
      src_tree_.reset(nullptr);
      trg_tree_.reset(nullptr);
      return false;
    }

    auto tree_height = fmm_tree_height(std::max(n_src_points_, n_fld_points_));

    auto tree_height_changed = tree_height_ != tree_height;
    tree_height_ = tree_height;

    if (!interpolator_ || tree_height_changed) {
      interpolator_ = std::make_unique<Interpolator>(kernel_, order_, tree_height, box_.width(0));
      far_field_ = std::make_unique<FarField>(*interpolator_);
      fmm_operator_ = std::make_unique<FmmOperator>(near_field_, *far_field_);
    }

    if (!src_tree_) {
      src_tree_ = std::make_unique<SourceTree>(tree_height, order_, box_, 10, 10, src_particles_);
      src_particles_.clear();
      src_particles_.shrink_to_fit();
      multipole_dirty_ = true;
    } else if (tree_height_changed) {
      std::vector<SourceParticle> particles(n_src_points_);

      scalfmm::component::for_each_leaf(std::begin(*src_tree_), std::end(*src_tree_),
                                        [&](auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename SourceLeaf::proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            auto& new_p = particles.at(idx);
                                            for (auto i = 0; i < 3; i++) {
                                              new_p.position(i) = p.position(i);
                                            }
                                            for (auto i = 0; i < Dim; i++) {
                                              new_p.inputs(i) = p.inputs(i);
                                            }
                                            new_p.variables(idx);
                                          }
                                        });

      src_tree_ = std::make_unique<SourceTree>(tree_height, order_, box_, 10, 10, particles);
      multipole_dirty_ = true;
    }

    if (!trg_tree_) {
      trg_tree_ = std::make_unique<TargetTree>(tree_height, order_, box_, 10, 10, trg_particles_);
      trg_particles_.clear();
      trg_particles_.shrink_to_fit();
    } else if (tree_height_changed) {
      std::vector<TargetParticle> particles(n_fld_points_);

      scalfmm::component::for_each_leaf(std::begin(*trg_tree_), std::end(*trg_tree_),
                                        [&](auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename TargetLeaf::proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            auto& new_p = particles.at(idx);
                                            for (auto i = 0; i < 3; i++) {
                                              new_p.position(i) = p.position(i);
                                            }
                                            new_p.variables(idx);
                                          }
                                        });

      trg_tree_ = std::make_unique<TargetTree>(tree_height, order_, box_, 10, 10, particles);
    }

    return true;
  }

  const model& model_;
  const Kernel kernel_;
  const int order_;
  const Box box_;
  const NearField near_field_;

  index_t n_src_points_{};
  index_t n_fld_points_{};
  mutable std::vector<SourceParticle> src_particles_;
  mutable std::vector<TargetParticle> trg_particles_;
  mutable bool multipole_dirty_{};
  mutable int tree_height_{};
  mutable std::unique_ptr<Interpolator> interpolator_;
  mutable std::unique_ptr<FarField> far_field_;
  mutable std::unique_ptr<FmmOperator> fmm_operator_;
  mutable std::unique_ptr<SourceTree> src_tree_;
  mutable std::unique_ptr<TargetTree> trg_tree_;
};

template <int Order, int Dim>
fmm_hessian_evaluator<Order, Dim>::fmm_hessian_evaluator(const model& model,
                                                         const geometry::bbox3d& bbox)
    : pimpl_(std::make_unique<impl>(model, bbox)) {}

template <int Order, int Dim>
fmm_hessian_evaluator<Order, Dim>::~fmm_hessian_evaluator() = default;

template <int Order, int Dim>
common::valuesd fmm_hessian_evaluator<Order, Dim>::evaluate() const {
  return pimpl_->evaluate();
}

template <int Order, int Dim>
void fmm_hessian_evaluator<Order, Dim>::set_field_points(const geometry::points3d& points) {
  pimpl_->set_field_points(points);
}

template <int Order, int Dim>
void fmm_hessian_evaluator<Order, Dim>::set_source_points(const geometry::points3d& points) {
  pimpl_->set_source_points(points);
}

template <int Order, int Dim>
void fmm_hessian_evaluator<Order, Dim>::set_weights(
    const Eigen::Ref<const common::valuesd>& weights) {
  pimpl_->set_weights(weights);
}

template class fmm_hessian_evaluator<6, 1>;
template class fmm_hessian_evaluator<10, 1>;
template class fmm_hessian_evaluator<6, 2>;
template class fmm_hessian_evaluator<10, 2>;
template class fmm_hessian_evaluator<6, 3>;
template class fmm_hessian_evaluator<10, 3>;

}  // namespace polatory::fmm
