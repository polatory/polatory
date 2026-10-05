#pragma once

#include <jizai/kriging/variogram_set.hpp>
#include <jizai/kriging/weight_function.hpp>
#include <jizai/model.hpp>
#include <memory>
#include <string>

namespace jizai::kriging {

template <int Dim>
class VariogramFitting {
  using Model = Model<Dim>;
  using VariogramSet = VariogramSet<Dim>;

 public:
  VariogramFitting(const VariogramSet& variog_set, const Model& model,
                   const WeightFunction& weight_fn = WeightFunction::kNumPairsOverDistanceSquared,
                   bool fit_anisotropy = true);

  ~VariogramFitting();

  VariogramFitting(const VariogramFitting&) = delete;
  VariogramFitting(VariogramFitting&&) = delete;
  VariogramFitting& operator=(const VariogramFitting&) = delete;
  VariogramFitting& operator=(VariogramFitting&&) = delete;

  std::string brief_report() const;

  double final_cost() const;

  std::string full_report() const;

  Model model() const;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace jizai::kriging
