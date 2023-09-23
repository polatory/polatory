#include <algorithm>
#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <scalfmm/algorithms/fmm.hpp>
#include <scalfmm/algorithms/full_direct.hpp>
#include <scalfmm/container/particle.hpp>
#include <scalfmm/interpolation/interpolation.hpp>
#include <scalfmm/matrix_kernels/laplace.hpp>
#include <scalfmm/meta/utils.hpp>
#include <scalfmm/operators/fmm_operators.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/cell.hpp>
#include <scalfmm/tree/for_each.hpp>
#include <scalfmm/tree/group_tree_view.hpp>
#include <scalfmm/tree/leaf_view.hpp>
#include <scalfmm/utils/accurater.hpp>
#include <vector>

#include "fmm_rbf_kernel2.hpp"

namespace polatory::fmm {

template <int Order>
class fmm_symmetric_evaluator<Order>::impl {
  using particle_type = scalfmm::container::particle<
      /* position */ double, 3,
      /* inputs */ double, 1,
      /* outputs */ double, 1,
      /* variables */ index_t>;

  using far_matrix_kernel_type = fmm_rbf_kernel2;
  using near_matrix_kernel_type = far_matrix_kernel_type;
  using near_field_type = scalfmm::operators::near_field_operator<near_matrix_kernel_type>;
  using interpolator_type =
      scalfmm::interpolation::interpolator<double, 3, far_matrix_kernel_type,
                                           scalfmm::options::uniform_<scalfmm::options::fft_>>;
  using far_field_type = scalfmm::operators::far_field_operator<interpolator_type>;
  using fmm_operator_type = scalfmm::operators::fmm_operators<near_field_type, far_field_type>;
  using cell_type = scalfmm::component::cell<typename interpolator_type::storage_type>;
  using leaf_type = scalfmm::component::leaf_view<particle_type>;
  using position_type = typename particle_type::position_type;
  using box_type = scalfmm::component::box<position_type>;
  using group_tree_type = scalfmm::component::group_tree_view<cell_type, leaf_type, box_type>;

 private:
  box_type make_box(const model& model, const geometry::bbox3d& bbox) {
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
        tree_height_(tree_height),
        box_(make_box(model, bbox)),
        near_field_(rbf_kernel_),
        interpolator_(rbf_kernel_, Order, tree_height, box_.width(0)),
        far_field_(interpolator_),
        fmm_operator_(near_field_, far_field_) {}

  common::valuesd evaluate() const {
    scalfmm::algorithms::fmm[scalfmm::options::_s(scalfmm::options::seq)](*tree_, fmm_operator_);

    auto self_potential = model_.rbf().evaluate_isotropic(geometry::vector3d::Zero());
    return potentials() + weights_ * self_potential;
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = points.rows();

    auto a = model_.rbf().anisotropy();

    particles_ = std::vector<particle_type>(n_points_);
    for (index_t i = 0; i < n_points_; i++) {
      auto& p = particles_.at(i);
      auto ap = geometry::transform_point(a, points.row(i));
      p.position() = position_type{ap(0), ap(1), ap(2)};
      p.inputs().at(0) = 0.0;
      p.outputs().at(0) = 0.0;
      p.variables(i);
    }

    tree_ = std::make_unique<group_tree_type>(tree_height_, Order, box_, 10, 10, particles_);
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    POLATORY_ASSERT(weights.rows() == n_points_);

    scalfmm::component::for_each_leaf(std::begin(*tree_), std::end(*tree_), [&](const auto& leaf) {
      // loop on the particles of the leaf
      for (auto p_ref : leaf) {
        // build a particle
        auto p = typename leaf_type::proxy_type(p_ref);
        auto idx = std::get<0>(p.variables());
        p.inputs().at(0).get() = weights(idx);
      }
    });

    weights_ = weights;
  }

 private:
  common::valuesd potentials() const {
    common::valuesd potentials(n_points_);

    scalfmm::component::for_each_leaf(
        std::cbegin(*tree_), std::cend(*tree_), [&](auto const& leaf) {
          // loop on the particles of the leaf
          for (auto const p_ref : leaf) {
            // build a particle
            const auto p = typename leaf_type::const_proxy_type(p_ref);
            auto idx = std::get<0>(p.variables());
            potentials(idx) = p.outputs().at(0);
          }
        });

    return potentials;
  }

  const model& model_;
  const far_matrix_kernel_type rbf_kernel_;
  const int tree_height_;

  index_t n_points_{};
  common::valuesd weights_;

  std::vector<particle_type> particles_;

  mutable box_type box_;
  mutable near_field_type near_field_;
  mutable interpolator_type interpolator_;
  mutable far_field_type far_field_;
  mutable fmm_operator_type fmm_operator_;
  mutable std::unique_ptr<group_tree_type> tree_;
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
