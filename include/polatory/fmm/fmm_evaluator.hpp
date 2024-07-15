#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/gradient_kernel.hpp>
#include <polatory/fmm/gradient_transpose_kernel.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <polatory/types.hpp>

namespace polatory::fmm {

template <int Dim>
class fmm_generic_evaluator_base {
  static constexpr int kDim = Dim;
  using Points = geometry::pointsNd<kDim>;

 public:
  fmm_generic_evaluator_base() = default;

  virtual ~fmm_generic_evaluator_base() = default;

  fmm_generic_evaluator_base(const fmm_generic_evaluator_base&) = delete;
  fmm_generic_evaluator_base(fmm_generic_evaluator_base&&) = delete;
  fmm_generic_evaluator_base& operator=(const fmm_generic_evaluator_base&) = delete;
  fmm_generic_evaluator_base& operator=(fmm_generic_evaluator_base&&) = delete;

  virtual vectord evaluate() const = 0;

  virtual void set_accuracy(double accuracy) = 0;

  virtual void set_source_points(const Points& points) = 0;

  virtual void set_target_points(const Points& points) = 0;

  virtual void set_weights(const Eigen::Ref<const vectord>& weights) = 0;
};

template <int Dim>
using FmmGenericEvaluatorPtr = std::unique_ptr<fmm_generic_evaluator_base<Dim>>;

template <class Rbf, class Kernel>
class fmm_generic_evaluator final : public fmm_generic_evaluator_base<Rbf::kDim> {
  static constexpr int kDim = Rbf::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  fmm_generic_evaluator(const Rbf& rbf, const Bbox& bbox);

  ~fmm_generic_evaluator() override;

  fmm_generic_evaluator(const fmm_generic_evaluator&) = delete;
  fmm_generic_evaluator(fmm_generic_evaluator&&) = delete;
  fmm_generic_evaluator& operator=(const fmm_generic_evaluator&) = delete;
  fmm_generic_evaluator& operator=(fmm_generic_evaluator&&) = delete;

  vectord evaluate() const override;

  void set_accuracy(double accuracy) override;

  void set_source_points(const Points& points) override;

  void set_target_points(const Points& points) override;

  void set_weights(const Eigen::Ref<const vectord>& weights) override;

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
FmmGenericEvaluatorPtr<Dim> make_fmm_evaluator(const rbf::rbf_proxy<Dim>& rbf,
                                               const geometry::bboxNd<Dim>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_evaluator(const rbf::rbf_proxy<Dim>& rbf,
                                                        const geometry::bboxNd<Dim>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_transpose_evaluator(
    const rbf::rbf_proxy<Dim>& rbf, const geometry::bboxNd<Dim>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_hessian_evaluator(const rbf::rbf_proxy<Dim>& rbf,
                                                       const geometry::bboxNd<Dim>& bbox);

}  // namespace polatory::fmm
