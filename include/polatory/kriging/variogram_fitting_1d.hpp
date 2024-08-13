#pragma once

#include <ceres/ceres.h>

#include <cmath>
#include <polatory/kriging/variogram.hpp>
#include <polatory/kriging/variogram_fitting.hpp>
#include <polatory/kriging/variogram_set.hpp>
#include <polatory/kriging/weight_function.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>
#include <string>
#include <thread>
#include <vector>

namespace polatory::kriging {

template <>
class VariogramFitting<1> {
  using Mat = Mat1;
  using Model = Model<1>;
  using Variogram = Variogram<1>;
  using VariogramSet = VariogramSet<1>;

 public:
  VariogramFitting(const VariogramSet& variog_set, const Model& model,
                   const WeightFunction& weight_fn = WeightFunction::kNumPairsOverDistanceSquared,
                   bool /*fit_anisotropy*/ = true)
      : model_template_(model),
        num_params_(model.num_parameters()),
        num_rbfs_(model.num_rbfs()),
        params_(model.parameters()) {
    for (auto& rbf : model_template_.rbfs()) {
      rbf.set_anisotropy(Mat::Identity());
    }

    ceres::Problem problem;

    problem.AddParameterBlock(params_.data(), num_params_);
    auto lbs = model.parameter_lower_bounds();
    auto ubs = model.parameter_upper_bounds();
    for (Index i = 0; i < num_params_; i++) {
      problem.SetParameterLowerBound(params_.data(), i, lbs.at(i));
      problem.SetParameterUpperBound(params_.data(), i, ubs.at(i));
    }

    for (const auto& variog : variog_set.variograms()) {
      auto* cost_fn = new ceres::DynamicNumericDiffCostFunction(
          new Residual(model_template_, variog, weight_fn));
      cost_fn->AddParameterBlock(num_params_);
      cost_fn->SetNumResiduals(variog.num_bins());
      problem.AddResidualBlock(cost_fn, nullptr, params_.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());

    Solve(options, &problem, &summary_);
  }

  std::string brief_report() const { return summary_.BriefReport(); }

  double final_cost() const { return summary_.final_cost; }

  std::string full_report() const { return summary_.FullReport(); }

  Model model() const {
    Model model{model_template_};
    model.set_parameters(params_);

    return model;
  }

 private:
  struct Residual {
    Residual(const Model& model_template, const Variogram& variog, const WeightFunction& weight_fn)
        : model_template_(model_template), variog_(variog), weight_fn_(weight_fn) {}

    bool operator()(const double* const* param_blocks, double* residuals) const {
      const auto* params = param_blocks[0];

      Model model{model_template_};

      std::vector<double> clamped_params(params, params + model.num_parameters());
      internal::clamp_parameters(clamped_params, model);
      model.set_parameters(clamped_params);

      return internal::compute_residuals(model, variog_, weight_fn_, residuals);
    }

   private:
    const Model& model_template_;
    const Variogram& variog_;
    const WeightFunction& weight_fn_;
  };

  Model model_template_;
  Index num_params_;
  Index num_rbfs_;
  std::vector<double> params_;
  ceres::Solver::Summary summary_;
};

}  // namespace polatory::kriging
