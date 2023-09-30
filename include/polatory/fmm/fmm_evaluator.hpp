#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/gradient_kernel.hpp>
#include <polatory/fmm/gradient_transpose_kernel.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/precision.hpp>

namespace polatory::fmm {

class fmm_base_evaluator {
 public:
  fmm_base_evaluator() = default;

  virtual ~fmm_base_evaluator() = default;

  fmm_base_evaluator(const fmm_base_evaluator&) = delete;
  fmm_base_evaluator(fmm_base_evaluator&&) = delete;
  fmm_base_evaluator& operator=(const fmm_base_evaluator&) = delete;
  fmm_base_evaluator& operator=(fmm_base_evaluator&&) = delete;

  virtual common::valuesd evaluate() const = 0;

  virtual void set_field_points(const geometry::points3d& points) = 0;

  virtual void set_source_points(const geometry::points3d& points) = 0;

  virtual void set_weights(const Eigen::Ref<const common::valuesd>& weights) = 0;
};

template <class Model, class Kernel>
class fmm_generic_evaluator : public fmm_base_evaluator {
 public:
  fmm_generic_evaluator(const Model& model, const geometry::bbox3d& bbox, precision prec);

  ~fmm_generic_evaluator() override;

  common::valuesd evaluate() const override;

  void set_field_points(const geometry::points3d& points) override;

  void set_source_points(const geometry::points3d& points) override;

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) override;

 private:
  class impl;

  std::unique_ptr<impl> impl_;
};

template <class Model, int Dim>
using fmm_evaluator = fmm_generic_evaluator<Model, kernel<typename Model::rbf_type, Dim>>;

template <class Model, int Dim>
using fmm_gradient_evaluator =
    fmm_generic_evaluator<Model, gradient_kernel<typename Model::rbf_type, Dim>>;

template <class Model, int Dim>
using fmm_gradient_transpose_evaluator =
    fmm_generic_evaluator<Model, gradient_transpose_kernel<typename Model::rbf_type, Dim>>;

template <class Model, int Dim>
using fmm_hessian_evaluator =
    fmm_generic_evaluator<Model, hessian_kernel<typename Model::rbf_type, Dim>>;

}  // namespace polatory::fmm
