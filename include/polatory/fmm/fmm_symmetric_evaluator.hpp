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

template <int Order, class Kernel>
class fmm_generic_symmetric_evaluator : public fmm_base_symmetric_evaluator {
 public:
  fmm_generic_symmetric_evaluator(const model& model, const geometry::bbox3d& bbox);

  ~fmm_generic_symmetric_evaluator() override;

  common::valuesd evaluate() const override;

  void set_points(const geometry::points3d& points) override;

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) override;

 private:
  class impl;

  std::unique_ptr<impl> pimpl_;
};

template <int Order, int Dim>
using fmm_symmetric_evaluator = fmm_generic_symmetric_evaluator<Order, kernel<Dim>>;

template <int Order, int Dim>
using fmm_hessian_symmetric_evaluator = fmm_generic_symmetric_evaluator<Order, hessian_kernel<Dim>>;

}  // namespace polatory::fmm
