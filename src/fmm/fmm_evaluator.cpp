// Copyright (c) 2016, GSI and The Polatory Authors.

#include <polatory/fmm/fmm_evaluator.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

#include <ScalFMM/Components/FTypedLeaf.hpp>
#include <ScalFMM/Containers/FOctree.hpp>
#include <ScalFMM/Core/FFmmAlgorithmThreadTsm.hpp>
#include <ScalFMM/Kernels/Chebyshev/FChebCell.hpp>
#include <ScalFMM/Kernels/Chebyshev/FChebSymKernel.hpp>
#include <ScalFMM/Kernels/P2P/FP2PParticleContainerIndexed.hpp>

#include "fmm_rbf_kernel.hpp"

namespace polatory {
namespace fmm {

template <int Order>
class fmm_evaluator<Order>::impl {
  using Cell = FTypedChebCell<double, Order>;
  using ParticleContainer = FP2PParticleContainerIndexed<double>;
  using Leaf = FTypedLeaf<double, ParticleContainer>;
  using Octree = FOctree<double, Cell, ParticleContainer, Leaf>;
  using InterpolatedKernel = FChebSymKernel<double, Cell, ParticleContainer, fmm_rbf_kernel, Order>;
  using Fmm = FFmmAlgorithmThreadTsm<Octree, Cell, ParticleContainer, InterpolatedKernel, Leaf>;

  static constexpr int FmmAlgorithmScheduleChunkSize = 1;

public:
  impl(const model& model, int tree_height, const geometry::bbox3d& bbox)
    : model_(model)
    , rbf_kernel_(model.rbf())
    , n_src_points_(0)
    , n_fld_points_(0) {
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
    tree_->forEachLeaf([&](Leaf* leaf) {
      auto& particles = *leaf->getTargets();
      particles.resetForcesAndPotential();
    });

    // clang-tidy complains for some reason if these executions are merged.
    fmm_->execute(FFmmM2L | FFmmL2L | FFmmL2P);
    fmm_->execute(FFmmP2P);

    return potentials();
  }

  void set_field_points(const geometry::points3d& points) {
    n_fld_points_ = points.rows();

    // Remove all target particles.
    tree_->forEachLeaf([&](Leaf* leaf) {
      auto& particles = *leaf->getTargets();
      particles.clear();
    });

    // Insert target particles.
    auto ti = model_.rbf().inverse_transformation();
    for (size_t idx = 0; idx < n_fld_points_; idx++) {
      auto ti_p = ti.transform_point(points.row(idx));
      tree_->insert(FPoint<double>(ti_p.data()), FParticleType::FParticleTypeTarget, idx, 0.0);
    }

    fmm_->updateTargetCells();

    update_potential_ptrs();
  }

  void set_source_points(const geometry::points3d& points) {
    n_src_points_ = points.rows();

    // Remove all source particles.
    tree_->forEachLeaf([&](Leaf* leaf) {
      auto& particles = *leaf->getSrc();
      particles.clear();
    });

    // Insert source particles.
    auto ti = model_.rbf().inverse_transformation();
    for (size_t idx = 0; idx < n_src_points_; idx++) {
      auto ti_p = ti.transform_point(points.row(idx));
      tree_->insert(FPoint<double>(ti_p.data()), FParticleType::FParticleTypeSource, idx, 0.0);
    }

    update_weight_ptrs();
  }

  void set_source_points_and_weights(const geometry::points3d& points, const Eigen::Ref<const common::valuesd>& weights) {
    assert(weights.rows() == points.rows());

    n_src_points_ = points.rows();

    // Remove all source particles.
    tree_->forEachLeaf([&](Leaf* leaf) {
      auto& particles = *leaf->getSrc();
      particles.clear();
    });

    // Insert source particles.
    auto ti = model_.rbf().inverse_transformation();
    for (size_t idx = 0; idx < n_src_points_; idx++) {
      auto ti_p = ti.transform_point(points.row(idx));
      tree_->insert(FPoint<double>(ti_p.data()), FParticleType::FParticleTypeSource, idx, weights[idx]);
    }

    tree_->forEachCell([&](Cell* cell) {
      cell->resetToInitialState();
    });

    fmm_->execute(FFmmP2M | FFmmM2M);

    weight_ptrs_.clear();
  }

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) {
    assert(weights.size() == n_src_points_);

    if (n_src_points_ == 0)
      return;

    if (weight_ptrs_.empty())
      update_weight_ptrs();

    for (size_t idx = 0; idx < n_src_points_; idx++) {
      *weight_ptrs_[idx] = weights[idx];
    }

    tree_->forEachCell([&](Cell* cell) {
      cell->resetToInitialState();
    });

    fmm_->execute(FFmmP2M | FFmmM2M);
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
    weight_ptrs_.resize(n_src_points_);
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

  size_t n_src_points_;
  size_t n_fld_points_;

  std::unique_ptr<Fmm> fmm_;
  std::unique_ptr<InterpolatedKernel> interpolated_kernel_;
  std::unique_ptr<Octree> tree_;

  std::vector<const double*> potential_ptrs_;
  std::vector<double*> weight_ptrs_;
};

template <int Order>
fmm_evaluator<Order>::fmm_evaluator(const model& model, int tree_height, const geometry::bbox3d& bbox)
  : pimpl_(std::make_unique<impl>(model, tree_height, bbox)) {
}

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
void fmm_evaluator<Order>::set_source_points_and_weights(const geometry::points3d& points, const Eigen::Ref<const common::valuesd>& weights) {
  pimpl_->set_source_points_and_weights(points, weights);
}

template <int Order>
void fmm_evaluator<Order>::set_weights(const Eigen::Ref<const common::valuesd>& weights) {
  pimpl_->set_weights(weights);
}

template class fmm_evaluator<6>;
template class fmm_evaluator<10>;

}  // namespace fmm
}  // namespace polatory
