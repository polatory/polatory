#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/types.hpp>
#include <polatory/fmm/gradient_kernel.hpp>
#include <polatory/fmm/gradient_transpose_kernel.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory::fmm {

template <class Rbf, class Kernel>
class fmm_generic_evaluator {
  static constexpr int kDim = Rbf::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  fmm_generic_evaluator(const Rbf& rbf, const Bbox& bbox, int order);

  ~fmm_generic_evaluator();

  common::valuesd evaluate() const;

  void set_source_points(const Points& points);

  void set_target_points(const Points& points);

  void set_weights(const Eigen::Ref<const common::valuesd>& weights);

 private:
  class impl;

  std::unique_ptr<impl> impl_;
};

template <class Rbf>
using fmm_evaluator = fmm_generic_evaluator<Rbf, kernel<Rbf>>;

template <class Rbf>
using fmm_gradient_evaluator = fmm_generic_evaluator<Rbf, gradient_kernel<Rbf>>;

template <class Rbf>
using fmm_gradient_transpose_evaluator = fmm_generic_evaluator<Rbf, gradient_transpose_kernel<Rbf>>;

template <class Rbf>
using fmm_hessian_evaluator = fmm_generic_evaluator<Rbf, hessian_kernel<Rbf>>;

}  // namespace polatory::fmm
