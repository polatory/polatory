#pragma once

#include <polatory/types.hpp>

namespace polatory::fmm {

template <class Container, class Kernel>
void full_direct(Container& particles, const Kernel& kernel) {
  static constexpr int km = static_cast<int>(Kernel::km);
  static constexpr int kn = static_cast<int>(Kernel::kn);
  auto n_points = static_cast<Index>(particles.size());

#pragma omp parallel for schedule(static)
  for (Index trg_idx = 0; trg_idx < n_points; trg_idx++) {
    auto p = particles.at(trg_idx);
    for (Index src_idx = 0; src_idx < n_points; src_idx++) {
      if (src_idx == trg_idx) {
        continue;
      }
      const auto q = particles.at(src_idx);
      auto k = kernel.evaluate(p.position(), q.position());
      for (auto i = 0; i < kn; i++) {
        for (auto j = 0; j < km; j++) {
          p.outputs(i) += q.inputs(j) * k.at(km * i + j);
        }
      }
    }
  }
}

template <class SourceContainer, class TargetContainer, class Kernel>
void full_direct(const SourceContainer& src_particles, TargetContainer& trg_particles,
                 const Kernel& kernel) {
  static constexpr int km = static_cast<int>(Kernel::km);
  static constexpr int kn = static_cast<int>(Kernel::kn);
  auto n_src_points = static_cast<Index>(src_particles.size());
  auto n_trg_points = static_cast<Index>(trg_particles.size());

#pragma omp parallel for schedule(static)
  for (Index trg_idx = 0; trg_idx < n_trg_points; trg_idx++) {
    auto p = trg_particles.at(trg_idx);
    for (Index src_idx = 0; src_idx < n_src_points; src_idx++) {
      const auto q = src_particles.at(src_idx);
      auto k = kernel.evaluate(p.position(), q.position());
      for (auto i = 0; i < kn; i++) {
        for (auto j = 0; j < km; j++) {
          p.outputs(i) += q.inputs(j) * k.at(km * i + j);
        }
      }
    }
  }
}

}  // namespace polatory::fmm
