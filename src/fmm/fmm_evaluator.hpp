#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/rbf/biharmonic2d.hpp>
#include <polatory/rbf/biharmonic3d.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/inverse_multiquadric1.hpp>
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
class fmm_generic_evaluator<Model, Kernel>::impl {
  static constexpr int kDim{Model::kDim};
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
  using Interpolator = scalfmm::interpolation::interpolator<
      double, kDim, Kernel,
      std::conditional_t<std::is_same_v<Kernel, kernel<typename Model::rbf_type>>,
                         scalfmm::options::chebyshev_<>, scalfmm::options::uniform_<>>>;
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
  impl(const Model& model, const Bbox& bbox, precision prec)
      : model_(model),
        kernel_(model.rbf()),
        order_(static_cast<int>(prec)),
        box_(make_box<Model, Box>(model, bbox)),
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
      if (!trg_tree_->is_interaction_m2l_lists_built()) {
        scalfmm::list::sequential::build_m2l_interaction_list(*src_tree_, *trg_tree_, 1);
      }
      if (!trg_tree_->is_interaction_p2p_lists_built()) {
        scalfmm::list::sequential::build_p2p_interaction_list(*src_tree_, *trg_tree_, 1, false);
      }
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::omp)](  //
          *src_tree_, *trg_tree_, *fmm_operator_, m2l | p2p);
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](  //
          *src_tree_, *trg_tree_, *fmm_operator_, l2l | l2p);
    }

    return potentials();
  }

  void set_source_points(const Points& points) {
    n_src_points_ = points.rows();

    src_particles_.resize(n_src_points_);

    auto a = model_.rbf().anisotropy();
    for (index_t idx = 0; idx < n_src_points_; idx++) {
      auto& p = src_particles_.at(idx);
      auto ap = geometry::transform_point<kDim>(a, points.row(idx));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
      }
      p.variables(idx);
    }

    src_tree_.reset(nullptr);
  }

  void set_target_points(const Points& points) {
    n_trg_points_ = points.rows();

    trg_particles_.resize(n_trg_points_);

    auto a = model_.rbf().anisotropy();
    for (index_t idx = 0; idx < n_trg_points_; idx++) {
      auto& p = trg_particles_.at(idx);
      auto ap = geometry::transform_point<kDim>(a, points.row(idx));
      for (auto i = 0; i < kDim; i++) {
        p.position(i) = ap(i);
      }
      p.variables(idx);
    }

    trg_tree_.reset(nullptr);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    POLATORY_ASSERT(weights.rows() == km * n_src_points_);

    if (!src_tree_) {
      for (index_t idx = 0; idx < n_src_points_; idx++) {
        auto& p = src_particles_.at(idx);
        for (auto i = 0; i < km; i++) {
          p.inputs(i) = weights(km * idx + i);
        }
      }
    } else {
      scalfmm::component::for_each_leaf(std::begin(*src_tree_), std::end(*src_tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename SourceLeaf::proxy_type(p_ref);
                                            auto idx = std::get<0>(p.variables());
                                            for (auto i = 0; i < km; i++) {
                                              p.inputs(i) = weights(km * idx + i);
                                            }
                                          }
                                        });
      multipole_dirty_ = true;
    }
  }

 private:
  common::valuesd potentials() const {
    common::valuesd potentials = common::valuesd::Zero(kn * n_trg_points_);

    if (trg_tree_) {
      scalfmm::component::for_each_leaf(std::cbegin(*trg_tree_), std::cend(*trg_tree_),
                                        [&](const auto& leaf) {
                                          for (auto p_ref : leaf) {
                                            auto p = typename TargetLeaf::const_proxy_type(p_ref);
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
    if (n_src_points_ == 0 || n_trg_points_ == 0) {
      interpolator_.reset(nullptr);
      far_field_.reset(nullptr);
      fmm_operator_.reset(nullptr);
      reset_src_tree();
      reset_trg_tree();
      tree_height_ = 0;
      return false;
    }

    auto tree_height = fmm_tree_height<kDim>(std::max(n_src_points_, n_trg_points_));
    if (tree_height_ != tree_height) {
      interpolator_ = std::make_unique<Interpolator>(kernel_, order_, tree_height, box_.width(0));
      far_field_ = std::make_unique<FarField>(*interpolator_);
      fmm_operator_ = std::make_unique<FmmOperator>(near_field_, *far_field_);
      reset_src_tree();
      reset_trg_tree();
      tree_height_ = tree_height;
    }

    if (!src_tree_) {
      src_tree_ = std::make_unique<SourceTree>(tree_height, order_, box_, 10, 10, src_particles_);
      src_particles_.clear();
      src_particles_.shrink_to_fit();
      multipole_dirty_ = true;
    }

    if (!trg_tree_) {
      trg_tree_ = std::make_unique<TargetTree>(tree_height, order_, box_, 10, 10, trg_particles_);
      trg_particles_.clear();
      trg_particles_.shrink_to_fit();
    }

    return true;
  }

  void reset_src_tree() const {
    if (!src_tree_) {
      return;
    }

    src_particles_.resize(n_src_points_);

    scalfmm::component::for_each_leaf(std::begin(*src_tree_), std::end(*src_tree_),
                                      [&](const auto& leaf) {
                                        for (auto p_ref : leaf) {
                                          auto p = typename SourceLeaf::proxy_type(p_ref);
                                          auto idx = std::get<0>(p.variables());
                                          auto& new_p = src_particles_.at(idx);
                                          for (auto i = 0; i < kDim; i++) {
                                            new_p.position(i) = p.position(i);
                                          }
                                          for (auto i = 0; i < km; i++) {
                                            new_p.inputs(i) = p.inputs(i);
                                          }
                                          new_p.variables(idx);
                                        }
                                      });

    src_tree_.reset(nullptr);
  }

  void reset_trg_tree() const {
    if (!trg_tree_) {
      return;
    }

    trg_particles_.resize(n_trg_points_);

    scalfmm::component::for_each_leaf(std::begin(*trg_tree_), std::end(*trg_tree_),
                                      [&](const auto& leaf) {
                                        for (auto p_ref : leaf) {
                                          auto p = typename TargetLeaf::proxy_type(p_ref);
                                          auto idx = std::get<0>(p.variables());
                                          auto& new_p = trg_particles_.at(idx);
                                          for (auto i = 0; i < kDim; i++) {
                                            new_p.position(i) = p.position(i);
                                          }
                                          new_p.variables(idx);
                                        }
                                      });

    trg_tree_.reset(nullptr);
  }

  const Model& model_;
  const Kernel kernel_;
  const int order_;
  const Box box_;
  const NearField near_field_;

  index_t n_src_points_{};
  index_t n_trg_points_{};
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

template <class Model, class Kernel>
fmm_generic_evaluator<Model, Kernel>::fmm_generic_evaluator(const Model& model, const Bbox& bbox,
                                                            precision prec)
    : impl_(std::make_unique<impl>(model, bbox, prec)) {}

template <class Model, class Kernel>
fmm_generic_evaluator<Model, Kernel>::~fmm_generic_evaluator() = default;

template <class Model, class Kernel>
common::valuesd fmm_generic_evaluator<Model, Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Model, class Kernel>
void fmm_generic_evaluator<Model, Kernel>::set_target_points(const Points& points) {
  impl_->set_target_points(points);
}

template <class Model, class Kernel>
void fmm_generic_evaluator<Model, Kernel>::set_source_points(const Points& points) {
  impl_->set_source_points(points);
}

template <class Model, class Kernel>
void fmm_generic_evaluator<Model, Kernel>::set_weights(
    const Eigen::Ref<const common::valuesd>& weights) {
  impl_->set_weights(weights);
}

#define IMPLEMENT_MODEL(MODEL)                                                               \
  template class fmm_generic_evaluator<MODEL, kernel<typename MODEL::rbf_type>>;             \
  template class fmm_generic_evaluator<MODEL, gradient_kernel<typename MODEL::rbf_type>>;    \
  template class fmm_generic_evaluator<MODEL,                                                \
                                       gradient_transpose_kernel<typename MODEL::rbf_type>>; \
  template class fmm_generic_evaluator<MODEL, hessian_kernel<typename MODEL::rbf_type>>;

#define IMPLEMENT_DIM(DIM)                                    \
  IMPLEMENT_MODEL(model<rbf::biharmonic2d<DIM>>);             \
  IMPLEMENT_MODEL(model<rbf::biharmonic3d<DIM>>);             \
  IMPLEMENT_MODEL(model<rbf::cov_exponential<DIM>>);          \
  IMPLEMENT_MODEL(model<rbf::cov_spheroidal3<DIM>>);          \
  IMPLEMENT_MODEL(model<rbf::cov_spheroidal5<DIM>>);          \
  IMPLEMENT_MODEL(model<rbf::cov_spheroidal7<DIM>>);          \
  IMPLEMENT_MODEL(model<rbf::cov_spheroidal9<DIM>>);          \
  IMPLEMENT_MODEL(model<rbf::inverse_multiquadric1<DIM>>);    \
  IMPLEMENT_MODEL(model<rbf::multiquadric1<DIM>>);            \
  IMPLEMENT_MODEL(model<rbf::reference::cov_gaussian<DIM>>);  \
  IMPLEMENT_MODEL(model<rbf::reference::cov_spherical<DIM>>); \
  IMPLEMENT_MODEL(model<rbf::reference::triharmonic3d<DIM>>);

}  // namespace polatory::fmm
