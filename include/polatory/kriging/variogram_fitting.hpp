#pragma once

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/kriging/weight_function.hpp>
#include <polatory/model.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace polatory::kriging {

template <int Dim>
class variogram_fitting {
  using EmpiricalVariogram = empirical_variogram<Dim>;
  using Model = model<Dim>;
  using Vector = geometry::vectorNd<Dim>;

 public:
  variogram_fitting(const EmpiricalVariogram& emp_variog, const Model& model,
                    const weight_function& weight_fn) {
    Model model2(model);

    auto num_params = model2.num_parameters();
    params_ = model2.parameters();

    const auto& bin_distance = emp_variog.bin_distance();
    const auto& bin_gamma = emp_variog.bin_gamma();
    const auto& bin_num_pairs = emp_variog.bin_num_pairs();
    auto num_bins = static_cast<index_t>(bin_num_pairs.size());

    if (num_bins < num_params) {
      throw std::invalid_argument(
          "The number of lags must be greater than or equal to the number of parameters (= " +
          std::to_string(num_params) + ").");
    }

    ceres::Problem problem;
    for (index_t i = 0; i < num_bins; i++) {
      if (bin_num_pairs.at(i) == 0) {
        continue;
      }

      auto* cost_fn = create_cost_function(&model2, bin_num_pairs.at(i), bin_distance.at(i),
                                           bin_gamma.at(i), weight_fn);
      problem.AddResidualBlock(cost_fn, nullptr, params_.data());
    }

    auto lower_bounds = model2.parameter_lower_bounds();
    auto upper_bounds = model2.parameter_upper_bounds();
    for (auto i = 0; i < num_params; i++) {
      problem.SetParameterLowerBound(params_.data(), i, lower_bounds.at(i));
      problem.SetParameterUpperBound(params_.data(), i, upper_bounds.at(i));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
  }

  const std::vector<double>& parameters() const { return params_; }

 private:
  struct residual {
    residual(Model* model, index_t num_pairs, double distance, double gamma,
             const weight_function& weight_fn)
        : model_(model),
          num_pairs_(num_pairs),
          distance_(distance),
          gamma_(gamma),
          weight_fn_(weight_fn) {}

    bool operator()(const double* const* param_blocks, double* residuals) const {
      const auto* params = param_blocks[0];

      std::vector<double> clamped_params(params, params + model_->num_parameters());
      auto lbs = model_->parameter_lower_bounds();
      auto ubs = model_->parameter_upper_bounds();
      for (auto i = 0; i < model_->num_parameters(); i++) {
        clamped_params.at(i) = std::clamp(clamped_params.at(i), lbs.at(i), ubs.at(i));
      }
      model_->set_parameters(clamped_params);

      auto model_gamma = model_->nugget();
      for (const auto& rbf : model_->rbfs()) {
        Vector v = Vector::Zero();
        v(0) = distance_;
        model_gamma += rbf.evaluate_isotropic(Vector::Zero()) - rbf.evaluate_isotropic(v);
      }

      residuals[0] = weight_fn_(num_pairs_, distance_, model_gamma) * (gamma_ - model_gamma);
      return !std::isnan(residuals[0]);
    }

   private:
    Model* model_;
    const index_t num_pairs_;
    const double distance_;
    const double gamma_;
    const weight_function& weight_fn_;
  };

  ceres::CostFunction* create_cost_function(Model* model, index_t num_pairs, double distance,
                                            double gamma, const weight_function& weight_fn) {
    auto* cost_fn = new ceres::DynamicNumericDiffCostFunction<residual>(
        new residual(model, num_pairs, distance, gamma, weight_fn));
    cost_fn->AddParameterBlock(model->num_parameters());
    cost_fn->SetNumResiduals(1);
    return cost_fn;
  }

  std::vector<double> params_;
};

}  // namespace polatory::kriging
