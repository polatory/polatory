// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/fmm/fmm_operator.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

#include <ScalFMM/Components/FSimpleLeaf.hpp>
#include <ScalFMM/Containers/FOctree.hpp>
#include <ScalFMM/Core/FFmmAlgorithmThread.hpp>
#include <ScalFMM/Kernels/Chebyshev/FChebCell.hpp>
#include <ScalFMM/Kernels/Chebyshev/FChebSymKernel.hpp>
#include <ScalFMM/Kernels/P2P/FP2PParticleContainerIndexed.hpp>

#include "fmm_rbf_kernel.hpp"

namespace polatory {
namespace fmm {

template <int Order>
class fmm_operator<Order>::impl {
  using Cell = FChebCell<double, Order>;
  using ParticleContainer = FP2PParticleContainerIndexed<double>;
  using Leaf = FSimpleLeaf<double, ParticleContainer>;
  using Octree = FOctree<double, Cell, ParticleContainer, Leaf>;
  using InterpolatedKernel = FChebSymKernel<double, Cell, ParticleContainer, fmm_rbf_kernel, Order>;
  using Fmm = FFmmAlgorithmThread<Octree, Cell, ParticleContainer, InterpolatedKernel, Leaf>;

  static constexpr int FmmAlgorithmScheduleChunkSize = 1;

public:
  impl(const model& model, int tree_height, const geometry::bbox3d& bbox)
    : model_(model)
    , rbf_kernel_(model.rbf())
    , n_points_(0) {
    auto ti_bbox = bbox.transform(model.rbf().inverse_transformation());
    auto width = (1.0 + 1.0 / 64.0) * ti_bbox.size().maxCoeff();
    auto center = ti_bbox.center();

    interpolated_kernel_ = std::make_unique<InterpolatedKernel>(
      tree_height, width, FPoint<double>(center.data()), &rbf_kernel_);

    tree_ = std::make_unique<Octree>(
      tree_height, std::max(1, tree_height - 4), width, FPoint<double>(center.data()));

    fmm_ = std::make_unique<Fmm>(tree_.get(), interpolated_kernel_.get(), static_cast<int>(FmmAlgorithmScheduleChunkSize));
  }

  common::valuesd evaluate() const {
    reset_tree();

    fmm_->execute();

    return potentials();
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = points.rows();

    // Remove all source particles.
    tree_->forEachLeaf([&](Leaf* leaf) {
      auto& particles = *leaf->getSrc();
      particles.clear();
    });

    // Insert points.
    auto ti = model_.rbf().inverse_transformation();
    for (size_t idx = 0; idx < n_points_; idx++) {
      auto ti_p = ti.transform_point(points.row(idx));
      tree_->insert(FPoint<double>(ti_p.data()), idx, 0.0);
    }

    update_weight_ptrs();
    update_potential_ptrs();
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    assert(weights.rows() == n_points_);

    // Update weights.
    for (size_t idx = 0; idx < n_points_; idx++) {
      *weight_ptrs_[idx] = weights[idx];
    }
  }

private:
  common::valuesd potentials() const {
    common::valuesd phi = common::valuesd::Zero(n_points_);

    for (size_t i = 0; i < n_points_; i++) {
      phi[i] = *potential_ptrs_[i];
    }

    return phi;
  }

  void reset_tree() const {
    tree_->forEachCell([&](Cell* cell) {
      cell->resetToInitialState();
    });

    tree_->forEachLeaf([&](Leaf* leaf) {
      auto& particles = *leaf->getTargets();
      particles.resetForcesAndPotential();
    });
  }

  void update_potential_ptrs() {
    potential_ptrs_.resize(n_points_);
    tree_->forEachLeaf([&](Leaf* leaf) {
      const auto& particles = *leaf->getTargets();

      const auto& indices = particles.getIndexes();
      const double* potentials = particles.getPotentials();

      const size_t n_particles = particles.getNbParticles();
      for (size_t i = 0; i < n_particles; i++) {
        const size_t idx = indices[i];

        potential_ptrs_[idx] = &potentials[i];
      }
    });
  }

  void update_weight_ptrs() {
    weight_ptrs_.resize(n_points_);
    tree_->forEachLeaf([&](Leaf* leaf) {
      auto& particles = *leaf->getSrc();

      const auto& indices = particles.getIndexes();
      double* weights = particles.getPhysicalValues();

      const size_t n_particles = particles.getNbParticles();
      for (size_t i = 0; i < n_particles; i++) {
        const size_t idx = indices[i];

        weight_ptrs_[idx] = &weights[i];
      }
    });
  }

  const model model_;
  const fmm_rbf_kernel rbf_kernel_;

  size_t n_points_;

  std::unique_ptr<Fmm> fmm_;
  std::unique_ptr<InterpolatedKernel> interpolated_kernel_;
  std::unique_ptr<Octree> tree_;

  std::vector<const double*> potential_ptrs_;
  std::vector<double*> weight_ptrs_;
};

template <int Order>
fmm_operator<Order>::fmm_operator(const model& model, int tree_height, const geometry::bbox3d& bbox)
  : pimpl_(std::make_unique<impl>(model, tree_height, bbox)) {
}

template <int Order>
fmm_operator<Order>::~fmm_operator() = default;

template <int Order>
common::valuesd fmm_operator<Order>::evaluate() const {
  return pimpl_->evaluate();
}

template <int Order>
void fmm_operator<Order>::set_points(const geometry::points3d& points) {
  pimpl_->set_points(points);
}

template <int Order>
void fmm_operator<Order>::set_weights(const Eigen::Ref<const common::valuesd>& weights) {
  pimpl_->set_weights(weights);
}

template class fmm_operator<6>;
template class fmm_operator<10>;

}  // namespace fmm
}  // namespace polatory
