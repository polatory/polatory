#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/model.hpp>
#include <polatory/types.hpp>

namespace polatory::fmm {

template <int Order, int Dim>
class fmm_hessian_symmetric_evaluator {
 public:
  fmm_hessian_symmetric_evaluator(const model& model, int tree_height,
                                  const geometry::bbox3d& bbox);

  ~fmm_hessian_symmetric_evaluator();

  fmm_hessian_symmetric_evaluator(const fmm_hessian_symmetric_evaluator&) = delete;
  fmm_hessian_symmetric_evaluator(fmm_hessian_symmetric_evaluator&&) = delete;
  fmm_hessian_symmetric_evaluator& operator=(const fmm_hessian_symmetric_evaluator&) = delete;
  fmm_hessian_symmetric_evaluator& operator=(fmm_hessian_symmetric_evaluator&&) = delete;

  common::valuesd evaluate() const;

  void set_points(const geometry::points3d& points);

  void set_weights(const Eigen::Ref<const common::valuesd>& weights);

 private:
  class impl;

  std::unique_ptr<impl> pimpl_;
};

}  // namespace polatory::fmm
