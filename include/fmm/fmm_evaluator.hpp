// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "../geometry/bbox3.hpp"
#include "../rbf/rbf_base.hpp"
#include "../third_party/ScalFMM/Components/FTypedLeaf.hpp"
#include "../third_party/ScalFMM/Containers/FOctree.hpp"
#include "../third_party/ScalFMM/Core/FFmmAlgorithmThreadTsm.hpp"
#include "../third_party/ScalFMM/Kernels/Chebyshev/FChebCell.hpp"
#include "../third_party/ScalFMM/Kernels/Chebyshev/FChebSymKernel.hpp"
#include "../third_party/ScalFMM/Kernels/P2P/FP2PParticleContainerIndexed.hpp"

namespace polatory {
namespace fmm {

template<int Order>
class fmm_evaluator {
   static const int FmmAlgorithmScheduleChunkSize = 1;
   using FReal = double;
   using Cell = FTypedChebCell<FReal, Order>;
   using ParticleContainer = FP2PParticleContainerIndexed<FReal>;
   using Leaf = FTypedLeaf<FReal, ParticleContainer>;
   using Octree = FOctree<FReal, Cell, ParticleContainer, Leaf>;
   using InterpolatedKernel = FChebSymKernel<FReal, Cell, ParticleContainer, rbf::rbf_base, Order>;
   using Fmm = FFmmAlgorithmThreadTsm<Octree, Cell, ParticleContainer, InterpolatedKernel, Leaf>;

   std::unique_ptr<Fmm> fmm;
   std::unique_ptr<InterpolatedKernel> interpolated_kernel;
   std::unique_ptr<Octree> tree;

   size_t n_src_points;
   size_t n_fld_points;

   std::vector<const FReal *> potential_ptrs;
   std::vector<FReal *> weight_ptrs;

   Eigen::VectorXd potentials() const
   {
      Eigen::VectorXd phi = Eigen::VectorXd::Zero(field_size());

      for (size_t i = 0; i < field_size(); i++) {
         phi[i] = *potential_ptrs[i];
      }

      return phi;
   }

   void update_potential_ptrs()
   {
      potential_ptrs.resize(field_size());
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

   void update_weight_ptrs()
   {
      weight_ptrs.resize(source_size());
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
   fmm_evaluator(const rbf::rbf_base& rbf, int tree_height, const geometry::bbox3d& bbox)
      : n_src_points(0)
      , n_fld_points(0)
   {
      auto bbox_width = (1.0 + 1.0 / 64.0) * (bbox.max - bbox.min).maxCoeff();
      Eigen::Vector3d bbox_center = (bbox.min + bbox.max) / 2.0;

      interpolated_kernel = std::make_unique<InterpolatedKernel>(
         tree_height, bbox_width, FPoint<FReal>(bbox_center.data()), &rbf);

      tree = std::make_unique<Octree>(
         tree_height, std::max(1, tree_height - 4), bbox_width, FPoint<FReal>(bbox_center.data()));

      fmm = std::make_unique<Fmm>(tree.get(), interpolated_kernel.get(), int(FmmAlgorithmScheduleChunkSize));
   }

   Eigen::VectorXd evaluate() const
   {
      tree->forEachLeaf([&](Leaf *leaf) {
         auto& particles = *leaf->getTargets();
         particles.resetForcesAndPotential();
      });

      fmm->execute(FFmmM2L | FFmmL2L | FFmmL2P | FFmmP2P);

      return potentials();
   }

   template<typename Container>
   void set_source_points(const Container& points)
   {
	  n_src_points = points.size();

      // Remove all source particles.
      tree->forEachLeaf([&](Leaf *leaf) {
         auto& particles = *leaf->getSrc();
         particles.clear();
      });

      // Insert source particles.
      for (size_t idx = 0; idx < points.size(); idx++) {
         tree->insert(FPoint<FReal>(points[idx].data()), FParticleType::FParticleTypeSource, idx, FReal(0));
      }

      update_weight_ptrs();
   }

   template<typename Container, typename Derived>
   void set_source_points_and_weights(const Container& points, const Eigen::MatrixBase<Derived>& weights)
   {
      assert(weights.size() == points.size());

	  n_src_points = points.size();

      // Remove all source particles.
      tree->forEachLeaf([&](Leaf *leaf) {
         auto& particles = *leaf->getSrc();
         particles.clear();
      });

      // Insert source particles.
      for (size_t idx = 0; idx < points.size(); idx++) {
         tree->insert(FPoint<FReal>(points[idx].data()), FParticleType::FParticleTypeSource, idx, weights[idx]);
      }

      tree->forEachCell([&](Cell *cell) {
         cell->resetToInitialState();
      });

      fmm->execute(FFmmP2M | FFmmM2M);

      weight_ptrs.clear();
   }

   template<typename Container>
   void set_field_points(const Container& points)
   {
      n_fld_points = points.size();

      // Remove all target particles.
      tree->forEachLeaf([&](Leaf *leaf) {
         auto& particles = *leaf->getTargets();
         particles.clear();
      });

      // Insert target particles.
      for (size_t idx = 0; idx < points.size(); idx++) {
         tree->insert(FPoint<FReal>(points[idx].data()), FParticleType::FParticleTypeTarget, idx, 0.0);
      }

      fmm->updateTargetCells();

      update_potential_ptrs();
   }

   template<typename Derived>
   void set_weights(const Eigen::MatrixBase<Derived>& weights)
   {
      assert(weights.size() == source_size());

      if (source_size() == 0)
         return;

      if (weight_ptrs.empty())
         update_weight_ptrs();

      for (size_t idx = 0; idx < source_size(); idx++) {
         *weight_ptrs[idx] = weights[idx];
      }

      tree->forEachCell([&](Cell *cell) {
         cell->resetToInitialState();
      });

      fmm->execute(FFmmP2M | FFmmM2M);
   }

   size_t source_size() const
   {
      return n_src_points;
   }

   size_t field_size() const
   {
      return n_fld_points;
   }
};

} // namespace fmm
} // namespace polatory
