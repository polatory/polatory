#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>

namespace polatory::fmm {

class fmm_base_symmetric_evaluator {
 public:
  fmm_base_symmetric_evaluator() = default;

  virtual ~fmm_base_symmetric_evaluator() = default;

  fmm_base_symmetric_evaluator(const fmm_base_symmetric_evaluator&) = delete;
  fmm_base_symmetric_evaluator(fmm_base_symmetric_evaluator&&) = delete;
  fmm_base_symmetric_evaluator& operator=(const fmm_base_symmetric_evaluator&) = delete;
  fmm_base_symmetric_evaluator& operator=(fmm_base_symmetric_evaluator&&) = delete;

  virtual common::valuesd evaluate() const = 0;

  virtual void set_points(const geometry::points3d& points) = 0;

  virtual void set_weights(const Eigen::Ref<const common::valuesd>& weights) = 0;
};

template <class Model, int Order, class Kernel>
class fmm_generic_symmetric_evaluator : public fmm_base_symmetric_evaluator {
 public:
  fmm_generic_symmetric_evaluator(const Model& model, const geometry::bbox3d& bbox);

  ~fmm_generic_symmetric_evaluator() override;

  common::valuesd evaluate() const override;

  void set_points(const geometry::points3d& points) override;

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) override;

 private:
  class impl;

  std::unique_ptr<impl> impl_;
};

template <class Model, int Order, int Dim>
using fmm_symmetric_evaluator =
    fmm_generic_symmetric_evaluator<Model, Order, kernel<typename Model::rbf_type, Dim>>;

template <class Model, int Order, int Dim>
using fmm_hessian_symmetric_evaluator =
    fmm_generic_symmetric_evaluator<Model, Order, hessian_kernel<typename Model::rbf_type, Dim>>;

}  // namespace polatory::fmm
