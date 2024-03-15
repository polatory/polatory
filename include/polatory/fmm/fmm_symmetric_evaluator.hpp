#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/common/types.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory::fmm {

template <class Rbf, class Kernel>
class fmm_generic_symmetric_evaluator {
  static constexpr int kDim = Rbf::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  fmm_generic_symmetric_evaluator(const Rbf& rbf, const Bbox& bbox, int order);

  ~fmm_generic_symmetric_evaluator();

  common::valuesd evaluate() const;

  void set_points(const Points& points);

  void set_weights(const Eigen::Ref<const common::valuesd>& weights);

 private:
  class impl;

  std::unique_ptr<impl> impl_;
};

template <class Rbf>
using fmm_symmetric_evaluator = fmm_generic_symmetric_evaluator<Rbf, kernel<Rbf>>;

template <class Rbf>
using fmm_hessian_symmetric_evaluator = fmm_generic_symmetric_evaluator<Rbf, hessian_kernel<Rbf>>;

}  // namespace polatory::fmm
