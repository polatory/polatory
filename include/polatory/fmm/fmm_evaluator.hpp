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
#include <polatory/third_party/ScalFMM/Components/FTypedLeaf.hpp>
#include <polatory/third_party/ScalFMM/Containers/FOctree.hpp>
#include <polatory/third_party/ScalFMM/Core/FFmmAlgorithmThreadTsm.hpp>
#include <polatory/third_party/ScalFMM/Kernels/Chebyshev/FChebCell.hpp>
#include <polatory/third_party/ScalFMM/Kernels/Chebyshev/FChebSymKernel.hpp>
#include <polatory/third_party/ScalFMM/Kernels/P2P/FP2PParticleContainerIndexed.hpp>

namespace polatory {
namespace fmm {

template <int Order>
class fmm_evaluator {
  static const int FmmAlgorithmScheduleChunkSize = 1;
  using FReal = double;
  using Cell = FTypedChebCell<FReal, Order>;
  using ParticleContainer = FP2PParticleContainerIndexed<FReal>;
  using Leaf = FTypedLeaf<FReal, ParticleContainer>;
  using Octree = FOctree<FReal, Cell, ParticleContainer, Leaf>;
  using InterpolatedKernel = FChebSymKernel<FReal, Cell, ParticleContainer, rbf::rbf_base, Order>;
  using Fmm = FFmmAlgorithmThreadTsm<Octree, Cell, ParticleContainer, InterpolatedKernel, Leaf>;

public:
  fmm_evaluator(const rbf::rbf_base& rbf, int tree_height, const geometry::bbox3d& bbox)
    : n_src_points_(0)
    , n_fld_points_(0) {
    auto bbox_width = (1.0 + 1.0 / 64.0) * bbox.size().maxCoeff();
    auto bbox_center = bbox.center();

    interpolated_kernel_ = std::make_unique<InterpolatedKernel>(
      tree_height, bbox_width, FPoint<FReal>(bbox_center.data()), &rbf);

    tree_ = std::make_unique<Octree>(
      tree_height, std::max(1, tree_height - 4), bbox_width, FPoint<FReal>(bbox_center.data()));

    fmm_ = std::make_unique<Fmm>(tree_.get(), interpolated_kernel_.get(), int(FmmAlgorithmScheduleChunkSize));
  }

  common::valuesd evaluate() const {
    tree_->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getTargets();
      particles.resetForcesAndPotential();
    });

    fmm_->execute(FFmmM2L | FFmmL2L | FFmmL2P | FFmmP2P);

    return potentials();
  }

  void set_source_points(const geometry::points3d& points) {
    n_src_points_ = points.rows();

    // Remove all source particles.
    tree_->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getSrc();
      particles.clear();
    });

    // Insert source particles.
    for (size_t idx = 0; idx < n_src_points_; idx++) {
      tree_->insert(FPoint<FReal>(points.row(idx).data()), FParticleType::FParticleTypeSource, idx, FReal(0));
    }

    update_weight_ptrs();
  }

  template <class Derived>
  void set_source_points_and_weights(const geometry::points3d& points, const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.rows() == points.rows());

    n_src_points_ = points.rows();

    // Remove all source particles.
    tree_->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getSrc();
      particles.clear();
    });

    // Insert source particles.
    for (size_t idx = 0; idx < n_src_points_; idx++) {
      tree_->insert(FPoint<FReal>(points.row(idx).data()), FParticleType::FParticleTypeSource, idx, weights[idx]);
    }

    tree_->forEachCell([&](Cell *cell) {
      cell->resetToInitialState();
    });

    fmm_->execute(FFmmP2M | FFmmM2M);

    weight_ptrs_.clear();
  }

  void set_field_points(const geometry::points3d& points) {
    n_fld_points_ = points.rows();

    // Remove all target particles.
    tree_->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getTargets();
      particles.clear();
    });

    // Insert target particles.
    for (size_t idx = 0; idx < n_fld_points_; idx++) {
      tree_->insert(FPoint<FReal>(points.row(idx).data()), FParticleType::FParticleTypeTarget, idx, 0.0);
    }

    fmm_->updateTargetCells();

    update_potential_ptrs();
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.size() == n_src_points_);

    if (source_size() == 0)
      return;

    if (weight_ptrs_.empty())
      update_weight_ptrs();

    for (size_t idx = 0; idx < n_src_points_; idx++) {
      *weight_ptrs_[idx] = weights[idx];
    }

    tree_->forEachCell([&](Cell *cell) {
      cell->resetToInitialState();
    });

    fmm_->execute(FFmmP2M | FFmmM2M);
  }

  size_t source_size() const {
    return n_src_points_;
  }

private:
  common::valuesd potentials() const {
    common::valuesd phi = common::valuesd::Zero(n_fld_points_);

    for (size_t i = 0; i < n_fld_points_; i++) {
      phi[i] = *potential_ptrs_[i];
    }

    return phi;
  }

  void update_potential_ptrs() {
    potential_ptrs_.resize(n_fld_points_);
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
    weight_ptrs_.resize(source_size());
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

  size_t n_src_points_;
  size_t n_fld_points_;

  std::vector<const FReal *> potential_ptrs_;
  std::vector<FReal *> weight_ptrs_;
};

} // namespace fmm
} // namespace polatory
