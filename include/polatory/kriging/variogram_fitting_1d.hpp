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
class variogram_fitting<1> {
  using Matrix = geometry::matrix1d;
  using Model = model<1>;
  using Variogram = variogram<1>;
  using VariogramSet = variogram_set<1>;

 public:
  variogram_fitting(
      const VariogramSet& variog_set, const Model& model,
      const weight_function& weight_fn = weight_function::kNumPairsOverDistanceSquared,
      bool /*fit_anisotropy*/ = true)
      : model_template_(model),
        num_params_(model.num_parameters()),
        num_rbfs_(model.num_rbfs()),
        params_(model.parameters()) {
    for (auto& rbf : model_template_.rbfs()) {
      rbf.set_anisotropy(Matrix::Identity());
    }

    ceres::Problem problem;

    problem.AddParameterBlock(params_.data(), num_params_);
    auto lbs = model.parameter_lower_bounds();
    auto ubs = model.parameter_upper_bounds();
    for (index_t i = 0; i < num_params_; i++) {
      problem.SetParameterLowerBound(params_.data(), i, lbs.at(i));
      problem.SetParameterUpperBound(params_.data(), i, ubs.at(i));
    }

    for (const auto& variog : variog_set.variograms()) {
      auto* cost_fn = new ceres::DynamicNumericDiffCostFunction(
          new residual(model_template_, variog, weight_fn));
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
  struct residual {
    residual(const Model& model_template, const Variogram& variog, const weight_function& weight_fn)
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
    const weight_function& weight_fn_;
  };

  Model model_template_;
  index_t num_params_;
  index_t num_rbfs_;
  std::vector<double> params_;
  ceres::Solver::Summary summary_;
};

}  // namespace polatory::kriging
