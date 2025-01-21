#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/gradient_kernel.hpp>
#include <polatory/fmm/gradient_transpose_kernel.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/types.hpp>

namespace polatory::fmm {

template <int Dim, int km, int kn>
class FmmGenericEvaluatorBase {
  static constexpr int kDim = Dim;
  using Points = geometry::Points<kDim>;

 public:
  FmmGenericEvaluatorBase() = default;

  virtual ~FmmGenericEvaluatorBase() = default;

  FmmGenericEvaluatorBase(const FmmGenericEvaluatorBase&) = delete;
  FmmGenericEvaluatorBase(FmmGenericEvaluatorBase&&) = delete;
  FmmGenericEvaluatorBase& operator=(const FmmGenericEvaluatorBase&) = delete;
  FmmGenericEvaluatorBase& operator=(FmmGenericEvaluatorBase&&) = delete;

  virtual VecX evaluate() const = 0;

  virtual void set_accuracy(double accuracy) = 0;

  virtual void set_source_points(const Points& points) = 0;

  virtual void set_target_points(const Points& points) = 0;

  virtual void set_weights(const Eigen::Ref<const VecX>& weights) = 0;
};

template <int Dim, int km, int kn>
using FmmGenericEvaluatorPtr = std::unique_ptr<FmmGenericEvaluatorBase<Dim, km, kn>>;

template <class Kernel>
class FmmGenericEvaluator final
    : public FmmGenericEvaluatorBase<Kernel::kDim, Kernel::km, Kernel::kn> {
  using Rbf = typename Kernel::Rbf;
  static constexpr int kDim = Kernel::kDim;

  using Bbox = geometry::Bbox<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  FmmGenericEvaluator(const Rbf& rbf, const Bbox& bbox);

  ~FmmGenericEvaluator() override;

  FmmGenericEvaluator(const FmmGenericEvaluator&) = delete;
  FmmGenericEvaluator(FmmGenericEvaluator&&) = delete;
  FmmGenericEvaluator& operator=(const FmmGenericEvaluator&) = delete;
  FmmGenericEvaluator& operator=(FmmGenericEvaluator&&) = delete;

  VecX evaluate() const override;

  void set_accuracy(double accuracy) override;

  void set_source_points(const Points& points) override;

  void set_target_points(const Points& points) override;

  void set_weights(const Eigen::Ref<const VecX>& weights) override;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

template <class Rbf>
using FmmEvaluator = FmmGenericEvaluator<Kernel<Rbf>>;

template <class Rbf>
using FmmGradientEvaluator = FmmGenericEvaluator<GradientKernel<Rbf>>;

template <class Rbf>
using FmmGradientTransposeEvaluator = FmmGenericEvaluator<GradientTransposeKernel<Rbf>>;

template <class Rbf>
using FmmHessianEvaluator = FmmGenericEvaluator<HessianKernel<Rbf>>;

template <int Dim>
FmmGenericEvaluatorPtr<Dim, 1, 1> make_fmm_evaluator(const rbf::Rbf<Dim>& rbf,
                                               const geometry::Bbox<Dim>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim, Dim, 1> make_fmm_gradient_evaluator(const rbf::Rbf<Dim>& rbf,
                                                        const geometry::Bbox<Dim>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim, 1, Dim> make_fmm_gradient_transpose_evaluator(const rbf::Rbf<Dim>& rbf,
                                                                  const geometry::Bbox<Dim>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim, Dim, Dim> make_fmm_hessian_evaluator(const rbf::Rbf<Dim>& rbf,
                                                       const geometry::Bbox<Dim>& bbox);

}  // namespace polatory::fmm
