#pragma once

#include <omp.h>

#include <algorithm>
#include <array>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <polatory/types.hpp>
#include <scalfmm/container/point.hpp>
#include <scalfmm/tree/box.hpp>
#include <scalfmm/tree/utils.hpp>
#include <vector>

namespace polatory::fmm {

template <int Dim>
class FmmTreeHeightEstimator {
  using Box = scalfmm::component::box<scalfmm::container::point<double, Dim>>;
  using CellCounts = boost::unordered_flat_map<std::size_t, Index>;

  static constexpr int kMaxLevel = 30 / Dim;

  struct InteractionCounts {
    Index far_field_interactions{};
    Index max_group_far_field_interactions{};
    Index max_group_near_field_pairs{};
    Index near_field_pairs{};
  };

 public:
  // Leaves and cells per group of the trees, the unit of ScalFMM's tasks.
  static constexpr int kGroupSize = 10;

  explicit FmmTreeHeightEstimator(const Box& box) : box_(box) {}

  template <class Container>
  void set_source_points(const Container& particles) {
    src_keys_ = morton_keys(particles);
    src_cells_.clear();
    level_counts_.reset();
  }

  template <class Container>
  void set_target_points(const Container& particles) {
    trg_keys_ = morton_keys(particles);
    std::ranges::sort(trg_keys_);
    level_counts_.reset();
  }

  int tree_height(int order, double m2l_product_cost_in_pairs) const {
    if (!level_counts_) {
      level_counts_ = level_counts();
    }

    auto n_threads = static_cast<double>(omp_get_max_threads());
    auto p = static_cast<double>(order);
    // The real-to-complex FFT halves the last dimension.
    auto product_cost =
        std::pow(2.0 * p - 1.0, static_cast<double>(Dim - 1)) * p * m2l_product_cost_in_pairs;
    auto best_level = 1;
    auto best_cost = std::numeric_limits<double>::infinity();
    // The M2L runs at every level down to the leaves.
    auto far_field_cost = 0.0;
    for (std::size_t i = 0; i < level_counts_->size(); i++) {
      const auto& counts = level_counts_->at(i);
      // Each pass is bounded below by its busiest group of target leaves or cells, which matters
      // for coarse trees with few groups.
      far_field_cost +=
          std::max(static_cast<double>(counts.far_field_interactions),
                   n_threads * static_cast<double>(counts.max_group_far_field_interactions)) *
          product_cost;
      auto cost = std::max(static_cast<double>(counts.near_field_pairs),
                           n_threads * static_cast<double>(counts.max_group_near_field_pairs)) +
                  far_field_cost;
      if (cost < best_cost) {
        best_cost = cost;
        best_level = static_cast<int>(i) + 1;
      }
    }

    return best_level + 1;
  }

 private:
  InteractionCounts interaction_counts(int level) const {
    auto shift = Dim * (kMaxLevel - level);
    auto n_cells = std::int64_t{1} << level;
    const auto& src_cells = source_cells(level);

    InteractionCounts counts;
    Index group_pairs{};
    Index group_interactions{};
    auto group_size = 0;
    auto flush_group = [&] {
      counts.max_group_near_field_pairs = std::max(counts.max_group_near_field_pairs, group_pairs);
      counts.max_group_far_field_interactions =
          std::max(counts.max_group_far_field_interactions, group_interactions);
      group_pairs = 0;
      group_interactions = 0;
      group_size = 0;
    };
    auto n_trg = static_cast<Index>(trg_keys_.size());
    Index i{};
    while (i < n_trg) {
      auto cell = trg_keys_.at(i) >> shift;
      Index n_trg_cell{};
      while (i < n_trg && (trg_keys_.at(i) >> shift) == cell) {
        n_trg_cell++;
        i++;
      }
      Index leaf_pairs{};
      Index leaf_interactions{};

      auto c = scalfmm::index::get_coordinate_from_morton_index<Dim>(cell);
      auto q = c;
      std::array<int, Dim> offset;
      offset.fill(-3);
      while (true) {
        auto adjacent = true;
        auto parents_adjacent = true;
        auto inside = true;
        for (auto k = 0; k < Dim; k++) {
          q.at(k) = c.at(k) + offset.at(k);
          adjacent = adjacent && std::abs(offset.at(k)) <= 1;
          parents_adjacent = parents_adjacent && std::abs((q.at(k) >> 1) - (c.at(k) >> 1)) <= 1;
          inside = inside && q.at(k) >= 0 && q.at(k) < n_cells;
        }
        if (inside && (adjacent || parents_adjacent)) {
          auto it = src_cells.find(scalfmm::index::get_morton_index(q));
          if (it != src_cells.end()) {
            if (adjacent) {
              leaf_pairs += n_trg_cell * it->second;
            } else {
              leaf_interactions++;
            }
          }
        }
        auto k = 0;
        while (k < Dim && ++offset.at(k) > 3) {
          offset.at(k) = -3;
          k++;
        }
        if (k == Dim) {
          break;
        }
      }

      counts.near_field_pairs += leaf_pairs;
      counts.far_field_interactions += leaf_interactions;
      group_pairs += leaf_pairs;
      group_interactions += leaf_interactions;
      if (++group_size == kGroupSize) {
        flush_group();
      }
    }
    flush_group();

    return counts;
  }

  static Index leaf_count(const std::vector<std::size_t>& sorted_keys, int shift) {
    auto n = static_cast<Index>(sorted_keys.size());
    Index count = n > 0 ? 1 : 0;
    for (Index i = 1; i < n; i++) {
      if ((sorted_keys.at(i) >> shift) != (sorted_keys.at(i - 1) >> shift)) {
        count++;
      }
    }
    return count;
  }

  std::vector<InteractionCounts> level_counts() const {
    std::vector<InteractionCounts> counts;
    auto n_trg = static_cast<Index>(trg_keys_.size());
    for (auto level = 1; level <= kMaxLevel; level++) {
      auto shift = Dim * (kMaxLevel - level);
      // Stop at the first level at which target leaves would average fewer than 8 points.
      if (!counts.empty() && n_trg < 8 * leaf_count(trg_keys_, shift)) {
        break;
      }
      counts.push_back(interaction_counts(level));
    }
    return counts;
  }

  template <class Container>
  std::vector<std::size_t> morton_keys(const Container& particles) const {
    auto n = static_cast<Index>(particles.size());
    std::vector<std::size_t> keys;
    keys.reserve(n);
    for (Index idx = 0; idx < n; idx++) {
      keys.push_back(
          scalfmm::index::get_morton_index(particles.at(idx).position(), box_, kMaxLevel));
    }
    return keys;
  }

  const CellCounts& source_cells(int level) const {
    while (static_cast<int>(src_cells_.size()) < level) {
      auto shift = Dim * (kMaxLevel - static_cast<int>(src_cells_.size()) - 1);
      CellCounts cells;
      for (auto key : src_keys_) {
        cells[key >> shift]++;
      }
      src_cells_.push_back(std::move(cells));
    }
    return src_cells_.at(level - 1);
  }

  const Box box_;
  mutable std::optional<std::vector<InteractionCounts>> level_counts_;
  mutable std::vector<CellCounts> src_cells_;
  std::vector<std::size_t> src_keys_;
  std::vector<std::size_t> trg_keys_;
};

}  // namespace polatory::fmm
