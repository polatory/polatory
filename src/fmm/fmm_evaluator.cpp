#include <memory>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>
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

#include "fmm_rbf_kernel2.hpp"

namespace polatory::fmm {

template <int Order>
class fmm_evaluator<Order>::impl {
  using SourceParticle = scalfmm::container::particle<
      /* position */ double, 3,
      /* inputs */ double, 1,
      /* outputs */ double, 1,  // should be 0
      /* variables */ index_t>;

  using TargetParticle = scalfmm::container::particle<
      /* position */ double, 3,
      /* inputs */ double, 1,  // should be 0
      /* outputs */ double, 1,
      /* variables */ index_t>;

  using NearField = scalfmm::operators::near_field_operator<fmm_rbf_kernel2>;
  using Interpolator = scalfmm::interpolation::interpolator<
      double, 3, fmm_rbf_kernel2, scalfmm::options::chebyshev_<scalfmm::options::low_rank_>>;
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
        rbf_kernel_(model.rbf()),
        order_(Order),
        tree_height_(tree_height),
        box_(make_box(model, bbox)),
        near_field_(rbf_kernel_, false),
        interpolator_(rbf_kernel_, order_, tree_height, box_.width(0)),
        far_field_(interpolator_),
        fmm_operator_(near_field_, far_field_) {}

  common::valuesd evaluate() const {
    using namespace scalfmm::algorithms;

    trg_tree_->reset_locals();
    trg_tree_->reset_outputs();

    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](
        *src_tree_, *trg_tree_, fmm_operator_, m2l | l2l | l2p | p2p);

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
      p.outputs().at(0) = 0.0;
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
      p.inputs().at(0) = 0.0;
      p.variables(i);
    }

    src_tree_ = std::make_unique<SourceTree>(tree_height_, order_, box_, 10, 10, particles);
  }

  void set_source_points_and_weights(const geometry::points3d& points,
                                     const Eigen::Ref<const common::valuesd>& weights) {
    using namespace scalfmm::algorithms;

    n_src_points_ = points.rows();

    auto a = model_.rbf().anisotropy();

    std::vector<SourceParticle> particles(n_src_points_);
    for (index_t i = 0; i < n_src_points_; i++) {
      auto& p = particles.at(i);
      auto ap = geometry::transform_point(a, points.row(i));
      p.position() = Position{ap(0), ap(1), ap(2)};
      p.inputs().at(0) = weights(i);
      p.variables(i);
    }

    src_tree_ = std::make_unique<SourceTree>(tree_height_, order_, box_, 10, 10, particles);

    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](*src_tree_, fmm_operator_,
                                                                          p2m | m2m);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    using namespace scalfmm::algorithms;

    POLATORY_ASSERT(weights.rows() == n_src_points_);

    scalfmm::component::for_each_leaf(std::begin(*src_tree_), std::end(*src_tree_),
                                      [&](const auto& leaf) {
                                        // loop on the particles of the leaf
                                        for (auto p_ref : leaf) {
                                          // build a particle
                                          auto p = typename SourceLeaf::proxy_type(p_ref);
                                          auto idx = std::get<0>(p.variables());
                                          p.inputs().at(0).get() = weights(idx);
                                        }
                                      });

    src_tree_->reset_multipoles();

    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](*src_tree_, fmm_operator_,
                                                                          p2m | m2m);
  }

 private:
  common::valuesd potentials() const {
    common::valuesd potentials(n_fld_points_);

    scalfmm::component::for_each_leaf(std::cbegin(*trg_tree_), std::cend(*trg_tree_),
                                      [&](const auto& leaf) {
                                        // loop on the particles of the leaf
                                        for (auto p_ref : leaf) {
                                          // build a particle
                                          auto p = typename TargetLeaf::const_proxy_type(p_ref);
                                          auto idx = std::get<0>(p.variables());
                                          potentials(idx) = p.outputs().at(0);
                                        }
                                      });

    return potentials;
  }

  const model& model_;
  const fmm_rbf_kernel2 rbf_kernel_;
  const int order_;
  const int tree_height_;

  index_t n_src_points_{};
  index_t n_fld_points_{};

  mutable Box box_;
  mutable NearField near_field_;
  mutable Interpolator interpolator_;
  mutable FarField far_field_;
  mutable FmmOperator fmm_operator_;
  mutable std::unique_ptr<SourceTree> src_tree_;
  mutable std::unique_ptr<TargetTree> trg_tree_;
};

template <int Order>
fmm_evaluator<Order>::fmm_evaluator(const model& model, int tree_height,
                                    const geometry::bbox3d& bbox)
    : pimpl_(std::make_unique<impl>(model, tree_height, bbox)) {}

template <int Order>
fmm_evaluator<Order>::~fmm_evaluator() = default;

template <int Order>
common::valuesd fmm_evaluator<Order>::evaluate() const {
  return pimpl_->evaluate();
}

template <int Order>
void fmm_evaluator<Order>::set_field_points(const geometry::points3d& points) {
  pimpl_->set_field_points(points);
}

template <int Order>
void fmm_evaluator<Order>::set_source_points(const geometry::points3d& points) {
  pimpl_->set_source_points(points);
}

template <int Order>
void fmm_evaluator<Order>::set_source_points_and_weights(
    const geometry::points3d& points, const Eigen::Ref<const common::valuesd>& weights) {
  pimpl_->set_source_points_and_weights(points, weights);
}

template <int Order>
void fmm_evaluator<Order>::set_weights(const Eigen::Ref<const common::valuesd>& weights) {
  pimpl_->set_weights(weights);
}

template class fmm_evaluator<6>;
template class fmm_evaluator<10>;

}  // namespace polatory::fmm
