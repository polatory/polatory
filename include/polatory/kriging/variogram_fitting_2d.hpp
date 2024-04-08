#pragma once

#include <ceres/ceres.h>

#include <Eigen/Geometry>
#include <cmath>
#include <numbers>
#include <polatory/geometry/point3d.hpp>
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
class variogram_fitting<2> {
  using Matrix = geometry::matrix2d;
  using Model = model<2>;
  using Variogram = variogram<2>;
  using VariogramSet = variogram_set<2>;

 public:
  variogram_fitting(
      const VariogramSet& variog_set, const Model& model,
      const weight_function& weight_fn = weight_function::kNumPairsOverDistanceSquared,
      bool fit_anisotropy = true)
      : model_template_(model),
        fit_anisotropy_(fit_anisotropy && variog_set.num_variograms() >= 2),
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

    if (fit_anisotropy_) {
      problem.AddParameterBlock(&r_.angle(), 1);

      inv_minor_.resize(num_rbfs_, 1.0);
      problem.AddParameterBlock(inv_minor_.data(), num_rbfs_);

      for (index_t i = 0; i < num_rbfs_; i++) {
        problem.SetParameterLowerBound(inv_minor_.data(), i, 1.0);
        problem.SetParameterUpperBound(inv_minor_.data(), i, 1e2);
      }
    }

    for (const auto& variog : variog_set.variograms()) {
      auto* cost_fn = new ceres::DynamicNumericDiffCostFunction(
          new residual(model_template_, variog, weight_fn, fit_anisotropy_));
      cost_fn->AddParameterBlock(num_params_);
      if (fit_anisotropy_) {
        cost_fn->AddParameterBlock(1);
        cost_fn->AddParameterBlock(num_rbfs_);
      }
      cost_fn->SetNumResiduals(variog.num_bins());
      if (fit_anisotropy_) {
        problem.AddResidualBlock(cost_fn, nullptr, params_.data(), &r_.angle(), inv_minor_.data());
      } else {
        problem.AddResidualBlock(cost_fn, nullptr, params_.data());
      }
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

    if (fit_anisotropy_) {
      Matrix inv_rot = r_.toRotationMatrix();
      for (index_t i = 0; i < num_rbfs_; i++) {
        auto& rbf = model.rbfs().at(i);

        Matrix inv_scale = Matrix::Identity();
        inv_scale(1, 1) = inv_minor_.at(i);
        Matrix aniso = inv_scale * inv_rot;

        rbf.set_anisotropy(aniso);
      }
    }

    return model;
  }

 private:
  struct residual {
    residual(const Model& model_template, const Variogram& variog, const weight_function& weight_fn,
             bool fit_anisotropy)
        : model_template_(model_template),
          variog_(variog),
          weight_fn_(weight_fn),
          fit_anisotropy_(fit_anisotropy) {}

    bool operator()(const double* const* param_blocks, double* residuals) const {
      const auto* params = param_blocks[0];

      Model model{model_template_};

      std::vector<double> clamped_params(params, params + model.num_parameters());
      internal::clamp_parameters(clamped_params, model);
      model.set_parameters(clamped_params);

      if (fit_anisotropy_) {
        const auto* angle = param_blocks[1];
        const auto* min_scale = param_blocks[2];

        Eigen::Rotation2Dd r(*angle);
        Matrix inv_rot = r.toRotationMatrix();
        auto num_rbfs = model.num_rbfs();
        for (index_t i = 0; i < num_rbfs; i++) {
          auto& rbf = model.rbfs().at(i);

          Matrix inv_scale = Matrix::Identity();
          inv_scale(1, 1) = min_scale[i];
          Matrix aniso = inv_scale * inv_rot;

          rbf.set_anisotropy(aniso);
        }
      }

      return internal::compute_residuals(model, variog_, weight_fn_, residuals);
    }

   private:
    const Model& model_template_;
    const Variogram& variog_;
    const weight_function& weight_fn_;
    bool fit_anisotropy_;
  };

  Model model_template_;
  bool fit_anisotropy_;
  index_t num_params_;
  index_t num_rbfs_;
  std::vector<double> params_;
  Eigen::Rotation2Dd r_{std::numbers::pi * Eigen::Matrix<double, 1, 1>::Random()(0)};
  std::vector<double> inv_minor_;
  ceres::Solver::Summary summary_;
};

}  // namespace polatory::kriging
