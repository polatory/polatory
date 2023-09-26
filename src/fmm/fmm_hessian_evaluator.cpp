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
        kernel_(model.rbf()),
        order_(Order + 2),
        tree_height_(tree_height),
        box_(make_box(model, bbox)),
        near_field_(kernel_, false),
        interpolator_(kernel_, order_, tree_height, box_.width(0)),
        far_field_(interpolator_),
        fmm_operator_(near_field_, far_field_) {}

  common::valuesd evaluate() const {
    using namespace scalfmm::algorithms;

    trg_tree_->reset_locals();
    trg_tree_->reset_outputs();

    // Prevent segfault.
    if (n_src_points_ > 0 && n_fld_points_ > 0) {
      scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](  //
          *src_tree_, *trg_tree_, fmm_operator_, m2l | l2l | l2p | p2p);
    }

    return potentials();
  }

  void set_field_points(const geometry::points3d& points) {
    n_fld_points_ = points.rows();

    auto a = model_.rbf().anisotropy();

    std::vector<TargetParticle> particles(n_fld_points_);
    for (index_t i = 0; i < n_fld_points_; i++) {
      auto& p = particles.at(i);
      auto ap = geometry::transform_point(a, points.row(i));
      p.position() = Position{ap(0), ap(1), ap(2)};
      p.variables(i);
    }

    trg_tree_ = std::make_unique<TargetTree>(tree_height_, order_, box_, 10, 10, particles);
  }

  void set_source_points(const geometry::points3d& points) {
    n_src_points_ = points.rows();

    auto a = model_.rbf().anisotropy();

    std::vector<SourceParticle> particles(n_src_points_);
    for (index_t i = 0; i < n_src_points_; i++) {
      auto& p = particles.at(i);
      auto ap = geometry::transform_point(a, points.row(i));
      p.position() = Position{ap(0), ap(1), ap(2)};
      p.variables(i);
    }

    src_tree_ = std::make_unique<SourceTree>(tree_height_, order_, box_, 10, 10, particles);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    using namespace scalfmm::algorithms;

    POLATORY_ASSERT(weights.rows() == Dim * n_src_points_);

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

    src_tree_->reset_multipoles();

    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](*src_tree_, fmm_operator_,
                                                                          p2m | m2m);
  }

 private:
  common::valuesd potentials() const {
    common::valuesd potentials(Dim * n_fld_points_);

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

    return potentials;
  }

  const model& model_;
  const Kernel kernel_;
  const int order_;
  const int tree_height_;

  index_t n_src_points_{};
  index_t n_fld_points_{};

  const Box box_;
  const NearField near_field_;
  const Interpolator interpolator_;
  const FarField far_field_;
  const FmmOperator fmm_operator_;
  mutable std::unique_ptr<SourceTree> src_tree_;
  mutable std::unique_ptr<TargetTree> trg_tree_;
};

template <int Order, int Dim>
fmm_hessian_evaluator<Order, Dim>::fmm_hessian_evaluator(const model& model, int tree_height,
                                                         const geometry::bbox3d& bbox)
    : pimpl_(std::make_unique<impl>(model, tree_height, bbox)) {}

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
