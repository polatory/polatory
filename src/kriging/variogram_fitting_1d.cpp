#include <ceres/ceres.h>

#include <polatory/kriging/variogram.hpp>
#include <polatory/kriging/variogram_fitting.hpp>
#include <polatory/types.hpp>
#include <string>
#include <thread>
#include <vector>

#include "variogram_fitting.hpp"

namespace polatory::kriging {

template <>
class VariogramFitting<1>::Impl {
  using Mat = Mat1;
  using Variogram = Variogram<1>;

 public:
  Impl(const VariogramSet& variog_set, const Model& model, const WeightFunction& weight_fn,
       bool /*fit_anisotropy*/)
      : model_template_(model),
        num_params_(static_cast<int>(model.num_parameters())),
        params_(model.parameters()) {
    for (auto& rbf : model_template_.rbfs()) {
      rbf.set_anisotropy(Mat::Identity());
    }

    ceres::Problem problem;

    problem.AddParameterBlock(params_.data(), num_params_);
    auto lbs = model.parameter_lower_bounds();
    auto ubs = model.parameter_upper_bounds();
    for (auto i = 0; i < num_params_; i++) {
      problem.SetParameterLowerBound(params_.data(), i, lbs.at(i));
      problem.SetParameterUpperBound(params_.data(), i, ubs.at(i));
    }

    for (const auto& variog : variog_set.variograms()) {
      auto* cost_fn = new ceres::DynamicNumericDiffCostFunction(
          new Residual(model_template_, variog, weight_fn));
      cost_fn->AddParameterBlock(num_params_);
      cost_fn->SetNumResiduals(static_cast<int>(variog.num_bins()));
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
      auto num_params = static_cast<int>(model.num_parameters());

      std::vector<double> clamped_params(params, params + num_params);
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
  int num_params_;
  std::vector<double> params_;
  ceres::Solver::Summary summary_;
};

template class VariogramFitting<1>;

}  // namespace polatory::kriging
