#pragma once

#include <ceres/ceres.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <polatory/geometry/point3d.hpp>
#include <polatory/kriging/variogram.hpp>
#include <polatory/kriging/weight_function.hpp>
#include <polatory/model.hpp>
#include <string>
#include <vector>

namespace polatory::kriging {

class vario_fitting {
  using Matrix = geometry::matrix3d;
  using Model = model<3>;
  using Variogram = variogram<3>;
  using Vector = geometry::vector3d;

 public:
  vario_fitting(const std::vector<Variogram>& variogs, const Model& model,
                const weight_function& weight_fn)
      : model_(model) {
    auto num_params = model_.num_parameters();
    params_ = model_.parameters();

    ceres::Problem problem;

    problem.AddParameterBlock(params_.data(), num_params);
    auto lower_bounds = model_.parameter_lower_bounds();
    auto upper_bounds = model_.parameter_upper_bounds();
    for (auto i = 0; i < num_params; i++) {
      problem.SetParameterLowerBound(params_.data(), i, lower_bounds.at(i));
      problem.SetParameterUpperBound(params_.data(), i, upper_bounds.at(i));
    }

    auto* quaternion_manifold = new ceres::EigenQuaternionManifold;
    problem.AddParameterBlock(q_.coeffs().data(), 4, quaternion_manifold);

    index_t num_rbfs = static_cast<index_t>(model_.rbfs().size());
    maj_scale_.resize(num_rbfs, 1.0);
    min_scale_.resize(num_rbfs, 1.0);
    problem.AddParameterBlock(maj_scale_.data(), num_rbfs);
    problem.AddParameterBlock(min_scale_.data(), num_rbfs);

    for (index_t i = 0; i < num_rbfs; i++) {
      problem.SetParameterLowerBound(maj_scale_.data(), i, 1e-2);
      problem.SetParameterUpperBound(maj_scale_.data(), i, 1.0);

      problem.SetParameterLowerBound(min_scale_.data(), i, 1.0);
      problem.SetParameterUpperBound(min_scale_.data(), i, 1e2);
    }

    for (const auto& variog : variogs) {
      auto* cost_fn = create_cost_function(&model_, variog, weight_fn);
      problem.AddResidualBlock(cost_fn, nullptr, params_.data(), q_.coeffs().data(),
                               maj_scale_.data(), min_scale_.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;

    Solve(options, &problem, &summary_);
    std::cout << summary_.BriefReport() << std::endl;
  }

  const Vector euler_angles() const { return q_.toRotationMatrix().eulerAngles(2, 0, 2); }

  Model model() const {
    Model model{model_};
    model.set_parameters(params_);

    index_t num_rbfs = static_cast<index_t>(model.rbfs().size());
    Eigen::Quaterniond q(q_.coeffs().data());
    Matrix rot = q.normalized().toRotationMatrix();
    for (index_t i = 0; i < num_rbfs; i++) {
      auto& rbf = model.rbfs().at(i);

      Eigen::Matrix3d scale = Eigen::Matrix3d::Identity();
      scale(0, 0) = maj_scale_.at(i);
      scale(2, 2) = min_scale_.at(i);
      Matrix aniso = scale * rot;

      rbf.set_anisotropy(aniso);
    }

    return model;
  }

  const std::vector<double>& parameters() const { return params_; }

  double final_cost() const { return summary_.final_cost; }

  const Matrix rotation() const { return q_.toRotationMatrix(); }

  const Vector scale(index_t i) const { return Vector{maj_scale_.at(i), 1.0, min_scale_.at(i)}; }

 private:
  struct residual {
    residual(Model* model, const Variogram& variog, const weight_function& weight_fn)
        : model_(model), variog_(variog), weight_fn_(weight_fn) {}

    bool operator()(const double* const* param_blocks, double* residuals) const {
      const auto* params = param_blocks[0];
      const auto* q_coeffs = param_blocks[1];
      const auto* maj_scale = param_blocks[2];
      const auto* min_scale = param_blocks[3];

      std::vector<double> clamped_params(params, params + model_->num_parameters());
      auto lbs = model_->parameter_lower_bounds();
      auto ubs = model_->parameter_upper_bounds();
      for (auto i = 0; i < model_->num_parameters(); i++) {
        clamped_params.at(i) = std::clamp(clamped_params.at(i), lbs.at(i), ubs.at(i));
      }
      model_->set_parameters(clamped_params);

      index_t num_rbfs = static_cast<index_t>(model_->rbfs().size());
      Eigen::Quaterniond q(q_coeffs);
      Matrix rot = q.normalized().toRotationMatrix();
      for (index_t i = 0; i < num_rbfs; i++) {
        auto& rbf = model_->rbfs().at(i);

        Eigen::Matrix3d scale = Eigen::Matrix3d::Identity();
        scale(0, 0) = maj_scale[i];
        scale(2, 2) = min_scale[i];
        Matrix aniso = scale * rot;

        rbf.set_anisotropy(aniso);
      }

      const Vector& dir = variog_.direction();

      auto num_bins = variog_.num_bins();
      auto total_weight = 0.0;
      for (index_t i = 0; i < num_bins; i++) {
        auto distance = variog_.bin_distance().at(i);
        auto gamma = variog_.bin_gamma().at(i);
        auto num_pairs = variog_.bin_num_pairs().at(i);

        auto model_gamma = model_->nugget();
        for (const auto& rbf : model_->rbfs()) {
          model_gamma += rbf.evaluate(Vector::Zero()) - rbf.evaluate(distance * dir);
        }

        auto weight = weight_fn_(distance, model_gamma, num_pairs);
        residuals[i] = weight * (gamma - model_gamma);
        if (std::isnan(residuals[i])) {
          return false;
        }

        total_weight += weight;
      }

      // Give the same total weight to each direction.
      for (index_t i = 0; i < num_bins; i++) {
        residuals[i] /= total_weight;
      }

      return true;
    }

   private:
    Model* model_;
    const Variogram& variog_;
    const weight_function& weight_fn_;
  };

  ceres::CostFunction* create_cost_function(Model* model, const Variogram& variog,
                                            const weight_function& weight_fn) {
    index_t num_rbfs = static_cast<index_t>(model->rbfs().size());
    auto* cost_fn =
        new ceres::DynamicNumericDiffCostFunction<residual>(new residual(model, variog, weight_fn));
    cost_fn->AddParameterBlock(model->num_parameters());
    cost_fn->AddParameterBlock(4);
    cost_fn->AddParameterBlock(num_rbfs);
    cost_fn->AddParameterBlock(num_rbfs);
    cost_fn->SetNumResiduals(variog.num_bins());
    return cost_fn;
  }

  Model model_;
  std::vector<double> params_;
  Eigen::Quaterniond q_{Eigen::Quaterniond::UnitRandom()};
  std::vector<double> maj_scale_;
  std::vector<double> min_scale_;
  ceres::Solver::Summary summary_;
};

}  // namespace polatory::kriging
