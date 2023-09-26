#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>

namespace polatory::fmm {

template <int Order, int Dim>
class fmm_gradient_evaluator {
 public:
  fmm_gradient_evaluator(const model& model, const geometry::bbox3d& bbox);

  ~fmm_gradient_evaluator();

  fmm_gradient_evaluator(const fmm_gradient_evaluator&) = delete;
  fmm_gradient_evaluator(fmm_gradient_evaluator&&) = delete;
  fmm_gradient_evaluator& operator=(const fmm_gradient_evaluator&) = delete;
  fmm_gradient_evaluator& operator=(fmm_gradient_evaluator&&) = delete;

  common::valuesd evaluate() const;

  void set_field_points(const geometry::points3d& points);

  void set_source_points(const geometry::points3d& points);

  void set_weights(const Eigen::Ref<const common::valuesd>& weights);

 private:
  class impl;

  std::unique_ptr<impl> pimpl_;
};

}  // namespace polatory::fmm
