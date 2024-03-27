#pragma once

#include <ceres/ceres.h>

#include <cmath>
#include <polatory/kriging/vario_fitting.hpp>
#include <thread>

namespace polatory::kriging {

template <>
class vario_fitting<1> {
  using Model = model<1>;
  using Variogram = variogram<1>;

 public:
  vario_fitting(const std::vector<Variogram>& variogs, const Model& model,
                const weight_function& weight_fn)
      : model_template_(model),
        num_params_(model.num_parameters()),
        num_rbfs_(model.num_rbfs()),
        params_(model.parameters()) {
    ceres::Problem problem;

    problem.AddParameterBlock(params_.data(), num_params_);
    auto lbs = model.parameter_lower_bounds();
    auto ubs = model.parameter_upper_bounds();
    for (auto i = 0; i < num_params_; i++) {
      problem.SetParameterLowerBound(params_.data(), i, lbs.at(i));
      problem.SetParameterUpperBound(params_.data(), i, ubs.at(i));
    }

    for (const auto& variog : variogs) {
      auto* cost_fn =
          new ceres::DynamicNumericDiffCostFunction(new residual(model, variog, weight_fn));
      cost_fn->AddParameterBlock(num_params_);
      cost_fn->SetNumResiduals(variog.num_lags());
      problem.AddResidualBlock(cost_fn, nullptr, params_.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.num_threads = std::thread::hardware_concurrency();

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

      auto num_lags = variog_.num_lags();
      for (index_t i = 0; i < num_lags; i++) {
        auto lag = variog_.bin_lag().at(i);
        auto gamma = variog_.bin_gamma().at(i);
        auto num_pairs = variog_.bin_num_pairs().at(i);

        auto model_gamma = internal::compute_model_gamma(model, lag);

        auto weight = weight_fn_(lag.norm(), model_gamma, num_pairs);
        residuals[i] = weight * (gamma - model_gamma);
        if (std::isnan(residuals[i])) {
          return false;
        }
      }

      return true;
    }

   private:
    const Model& model_template_;
    const Variogram& variog_;
    const weight_function& weight_fn_;
  };

  const Model& model_template_;
  index_t num_params_;
  index_t num_rbfs_;
  std::vector<double> params_;
  ceres::Solver::Summary summary_;
};

}  // namespace polatory::kriging
