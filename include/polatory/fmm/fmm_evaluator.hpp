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
#include <polatory/rbf/cov_spherical.hpp>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory::fmm {

template <int Dim>
class fmm_generic_evaluator_base {
  static constexpr int kDim = Dim;
  using Points = geometry::pointsNd<kDim>;

 public:
  virtual ~fmm_generic_evaluator_base() = default;

  virtual common::valuesd evaluate() const = 0;

  virtual void set_source_points(const Points& points) = 0;

  virtual void set_target_points(const Points& points) = 0;

  virtual void set_weights(const Eigen::Ref<const common::valuesd>& weights) = 0;
};

template <int Dim>
using FmmGenericEvaluatorPtr = std::unique_ptr<fmm_generic_evaluator_base<Dim>>;

template <class Rbf, class Kernel>
class fmm_generic_evaluator : public fmm_generic_evaluator_base<Rbf::kDim> {
  static constexpr int kDim = Rbf::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  fmm_generic_evaluator(const Rbf& rbf, const Bbox& bbox, int order);

  ~fmm_generic_evaluator() override;

  common::valuesd evaluate() const override;

  void set_source_points(const Points& points) override;

  void set_target_points(const Points& points) override;

  void set_weights(const Eigen::Ref<const common::valuesd>& weights) override;

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

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_evaluator(const rbf::RbfPtr<Dim>& rbf,
                                               const geometry::bboxNd<Dim>& bbox, int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_evaluator(const rbf::RbfPtr<Dim>& rbf,
                                                        const geometry::bboxNd<Dim>& bbox,
                                                        int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_transpose_evaluator(const rbf::RbfPtr<Dim>& rbf,
                                                                  const geometry::bboxNd<Dim>& bbox,
                                                                  int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_hessian_evaluator(const rbf::RbfPtr<Dim>& rbf,
                                                       const geometry::bboxNd<Dim>& bbox,
                                                       int order);

}  // namespace polatory::fmm
