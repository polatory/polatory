// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include <polatory/common/types.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/third_party/ScalFMM/Components/FSimpleLeaf.hpp>
#include <polatory/third_party/ScalFMM/Containers/FOctree.hpp>
#include <polatory/third_party/ScalFMM/Core/FFmmAlgorithmThread.hpp>
#include <polatory/third_party/ScalFMM/Kernels/Chebyshev/FChebCell.hpp>
#include <polatory/third_party/ScalFMM/Kernels/Chebyshev/FChebSymKernel.hpp>
#include <polatory/third_party/ScalFMM/Kernels/P2P/FP2PParticleContainerIndexed.hpp>

namespace polatory {
namespace fmm {

template <int Order>
class fmm_operator {
  static const int FmmAlgorithmScheduleChunkSize = 1;
  using FReal = double;
  using Cell = FChebCell<FReal, Order>;
  using ParticleContainer = FP2PParticleContainerIndexed<FReal>;
  using Leaf = FSimpleLeaf<FReal, ParticleContainer>;
  using Octree = FOctree<FReal, Cell, ParticleContainer, Leaf>;
  using InterpolatedKernel = FChebSymKernel<FReal, Cell, ParticleContainer, rbf::rbf_base, Order>;
  using Fmm = FFmmAlgorithmThread<Octree, Cell, ParticleContainer, InterpolatedKernel, Leaf>;

public:
  fmm_operator(const rbf::rbf_base& rbf, int tree_height, const geometry::bbox3d& bbox)
    : n_points_(0) {
    auto bbox_width = (1.0 + 1.0 / 64.0) * bbox.size().maxCoeff();
    auto bbox_center = bbox.center();

    interpolated_kernel_ = std::make_unique<InterpolatedKernel>(
      tree_height, bbox_width, FPoint<FReal>(bbox_center.data()), &rbf);

    tree_ = std::make_unique<Octree>(
      tree_height, std::max(1, tree_height - 4), bbox_width, FPoint<FReal>(bbox_center.data()));

    fmm_ = std::make_unique<Fmm>(tree_.get(), interpolated_kernel_.get(), int(FmmAlgorithmScheduleChunkSize));
  }

  common::valuesd evaluate() const {
    reset_tree();

    fmm_->execute();

    return potentials();
  }

  void set_points(const geometry::points3d& points) {
    n_points_ = points.rows();

    // Remove all source particles.
    tree_->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getSrc();
      particles.clear();
    });

    // Insert points.
    for (size_t idx = 0; idx < n_points_; idx++) {
      tree_->insert(FPoint<FReal>(points.row(idx).data()), idx, FReal(0));
    }

    update_weight_ptrs();
    update_potential_ptrs();
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.rows() == n_points_);

    // Update weights.
    for (size_t idx = 0; idx < n_points_; idx++) {
      *weight_ptrs_[idx] = weights[idx];
    }
  }

  size_t size() const {
    return n_points_;
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
    tree_->forEachCell([&](Cell *cell) {
      cell->resetToInitialState();
    });

    tree_->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getTargets();
      particles.resetForcesAndPotential();
    });
  }

  void update_potential_ptrs() {
    potential_ptrs_.resize(n_points_);
    tree_->forEachLeaf([&](Leaf *leaf) {
      const auto& particles = *leaf->getTargets();

      const auto& indices = particles.getIndexes();
      const FReal *potentials = particles.getPotentials();

      const size_t n_particles = particles.getNbParticles();
      for (size_t i = 0; i < n_particles; i++) {
        const size_t idx = indices[i];

        potential_ptrs_[idx] = &potentials[i];
      }
    });
  }

  void update_weight_ptrs() {
    weight_ptrs_.resize(n_points_);
    tree_->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getSrc();

      const auto& indices = particles.getIndexes();
      FReal *weights = particles.getPhysicalValues();

      const size_t n_particles = particles.getNbParticles();
      for (size_t i = 0; i < n_particles; i++) {
        const size_t idx = indices[i];

        weight_ptrs_[idx] = &weights[i];
      }
    });
  }

  std::unique_ptr<Fmm> fmm_;
  std::unique_ptr<InterpolatedKernel> interpolated_kernel_;
  std::unique_ptr<Octree> tree_;

  size_t n_points_;

  std::vector<const FReal *> potential_ptrs_;
  std::vector<FReal *> weight_ptrs_;
};

} // namespace fmm
} // namespace polatory
