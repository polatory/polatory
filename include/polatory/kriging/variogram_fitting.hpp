#pragma once

#include <ceres/ceres.h>

#include <iostream>
#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/kriging/weight_function.hpp>
#include <polatory/model.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory::kriging {

template <class Model>
class variogram_fitting {
 public:
  variogram_fitting(const empirical_variogram& emp_variog, const Model& model,
                    const weight_function& weight_fn) {
    Model model2(model);

    auto n_params = model2.num_parameters();
    params_ = model2.parameters();

    const auto& bin_distance = emp_variog.bin_distance();
    const auto& bin_gamma = emp_variog.bin_gamma();
    const auto& bin_num_pairs = emp_variog.bin_num_pairs();
    auto n_bins = static_cast<index_t>(bin_num_pairs.size());

    if (n_bins < n_params) {
      throw std::invalid_argument(
          "The number of lags must be greater than or equal to the number of parameters (= " +
          std::to_string(n_params) + ").");
    }

    ceres::Problem problem;
    for (index_t i = 0; i < n_bins; i++) {
      if (bin_num_pairs.at(i) == 0) {
        continue;
      }

      auto* cost_fn = create_cost_function(&model2, bin_num_pairs.at(i), bin_distance.at(i),
                                           bin_gamma.at(i), weight_fn);
      problem.AddResidualBlock(cost_fn, nullptr, params_.data());
    }

    auto lower_bounds = model2.parameter_lower_bounds();
    auto upper_bounds = model2.parameter_upper_bounds();
    for (auto i = 0; i < n_params; i++) {
      problem.SetParameterLowerBound(params_.data(), i, lower_bounds.at(i));
      problem.SetParameterUpperBound(params_.data(), i, upper_bounds.at(i));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 32;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
  }

  const std::vector<double>& parameters() const { return params_; }

 private:
  struct residual {
    residual(Model* model, index_t n_pairs, double distance, double gamma,
             const weight_function& weight_fn)
        : model_(model),
          n_pairs_(n_pairs),
          distance_(distance),
          gamma_(gamma),
          weight_fn_(weight_fn) {}

    bool operator()(const double* const* param_blocks, double* residuals) const {
      const auto* params = param_blocks[0];
      auto sill = params[0] + params[1];
      model_->set_parameters(std::vector<double>(params, params + model_->num_parameters()));
      auto model_gamma =
          sill - model_->rbf().evaluate_isotropic(geometry::vector3d{distance_, 0.0, 0.0});
      residuals[0] = weight_fn_(n_pairs_, distance_, model_gamma) * (gamma_ - model_gamma);

      return true;
    }

   private:
    Model* model_;
    const index_t n_pairs_;
    const double distance_;
    const double gamma_;
    const weight_function& weight_fn_;
  };

  ceres::CostFunction* create_cost_function(Model* model, index_t n_pairs, double distance,
                                            double gamma, const weight_function& weight_fn) {
    auto* cost_fn = new ceres::DynamicNumericDiffCostFunction<residual>(
        new residual(model, n_pairs, distance, gamma, weight_fn));
    cost_fn->AddParameterBlock(model->num_parameters());
    cost_fn->SetNumResiduals(1);
    return cost_fn;
  }

  std::vector<double> params_;
};

}  // namespace polatory::kriging
