// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include <polatory/geometry/bbox3d.hpp>
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

  std::unique_ptr<Fmm> fmm;
  std::unique_ptr<InterpolatedKernel> interpolated_kernel;
  std::unique_ptr<Octree> tree;

  size_t n_points;

  std::vector<const FReal *> potential_ptrs;
  std::vector<FReal *> weight_ptrs;

  Eigen::VectorXd potentials() const {
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(size());

    for (size_t i = 0; i < size(); i++) {
      phi[i] = *potential_ptrs[i];
    }

    return phi;
  }

  void reset_tree() const {
    tree->forEachCell([&](Cell *cell) {
      cell->resetToInitialState();
    });

    tree->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getTargets();
      particles.resetForcesAndPotential();
    });
  }

  void update_potential_ptrs() {
    potential_ptrs.resize(size());
    tree->forEachLeaf([&](Leaf *leaf) {
      const auto& particles = *leaf->getTargets();

      const auto& indices = particles.getIndexes();
      const FReal *potentials = particles.getPotentials();

      const size_t n_particles = particles.getNbParticles();
      for (size_t i = 0; i < n_particles; i++) {
        const size_t idx = indices[i];

        potential_ptrs[idx] = &potentials[i];
      }
    });
  }

  void update_weight_ptrs() {
    weight_ptrs.resize(size());
    tree->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getSrc();

      const auto& indices = particles.getIndexes();
      FReal *weights = particles.getPhysicalValues();

      const size_t n_particles = particles.getNbParticles();
      for (size_t i = 0; i < n_particles; i++) {
        const size_t idx = indices[i];

        weight_ptrs[idx] = &weights[i];
      }
    });
  }

public:
  fmm_operator(const rbf::rbf_base& rbf, int tree_height, const geometry::bbox3d& bbox)
    : n_points(0) {
    auto bbox_width = (1.0 + 1.0 / 64.0) * bbox.size().maxCoeff();
    auto bbox_center = bbox.center();

    interpolated_kernel = std::make_unique<InterpolatedKernel>(
      tree_height, bbox_width, FPoint<FReal>(bbox_center.data()), &rbf);

    tree = std::make_unique<Octree>(
      tree_height, std::max(1, tree_height - 4), bbox_width, FPoint<FReal>(bbox_center.data()));

    fmm = std::make_unique<Fmm>(tree.get(), interpolated_kernel.get(), int(FmmAlgorithmScheduleChunkSize));
  }

  Eigen::VectorXd evaluate() const {
    reset_tree();

    fmm->execute();

    return potentials();
  }

  template <class Container>
  void set_points(const Container& points) {
    n_points = points.size();

    // Remove all source particles.
    tree->forEachLeaf([&](Leaf *leaf) {
      auto& particles = *leaf->getSrc();
      particles.clear();
    });

    // Insert points.
    for (size_t idx = 0; idx < points.size(); idx++) {
      tree->insert(FPoint<FReal>(points[idx].data()), idx, FReal(0));
    }

    update_weight_ptrs();
    update_potential_ptrs();
  }

  template <class Derived>
  void set_weights(const Eigen::MatrixBase<Derived>& weights) {
    assert(weights.size() == size());

    // Update weights.
    for (size_t idx = 0; idx < size(); idx++) {
      *weight_ptrs[idx] = weights[idx];
    }
  }

  size_t size() const {
    return n_points;
  }
};

} // namespace fmm
} // namespace polatory
