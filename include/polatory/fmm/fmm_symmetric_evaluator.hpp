#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/types.hpp>

namespace polatory::fmm {

template <int Dim, int kn>
class FmmGenericSymmetricEvaluatorBase {
  static constexpr int kDim = Dim;
  using Points = geometry::Points<kDim>;
  using Resource = Resource<kDim, kn>;

 public:
  FmmGenericSymmetricEvaluatorBase() = default;

  virtual ~FmmGenericSymmetricEvaluatorBase() = default;

  FmmGenericSymmetricEvaluatorBase(const FmmGenericSymmetricEvaluatorBase&) = delete;
  FmmGenericSymmetricEvaluatorBase(FmmGenericSymmetricEvaluatorBase&&) = delete;
  FmmGenericSymmetricEvaluatorBase& operator=(const FmmGenericSymmetricEvaluatorBase&) = delete;
  FmmGenericSymmetricEvaluatorBase& operator=(FmmGenericSymmetricEvaluatorBase&&) = delete;

  virtual VecX evaluate() const = 0;

  virtual void set_accuracy(double accuracy) = 0;

  virtual void set_resource(const Resource& resource) = 0;
};

template <int Dim, int kn>
using FmmGenericSymmetricEvaluatorPtr = std::unique_ptr<FmmGenericSymmetricEvaluatorBase<Dim, kn>>;

template <class Kernel>
class FmmGenericSymmetricEvaluator final
    : public FmmGenericSymmetricEvaluatorBase<Kernel::kDim, Kernel::kn> {
  using Rbf = typename Kernel::Rbf;
  static constexpr int kDim = Kernel::kDim;

  using Bbox = geometry::Bbox<kDim>;
  using Points = geometry::Points<kDim>;
  using Resource = Resource<kDim, Kernel::kn>;

 public:
  FmmGenericSymmetricEvaluator(const Rbf& rbf, const Bbox& bbox);

  ~FmmGenericSymmetricEvaluator() override;

  FmmGenericSymmetricEvaluator(const FmmGenericSymmetricEvaluator&) = delete;
  FmmGenericSymmetricEvaluator(FmmGenericSymmetricEvaluator&&) = delete;
  FmmGenericSymmetricEvaluator& operator=(const FmmGenericSymmetricEvaluator&) = delete;
  FmmGenericSymmetricEvaluator& operator=(FmmGenericSymmetricEvaluator&&) = delete;

  VecX evaluate() const override;

  void set_accuracy(double accuracy) override;

  void set_resource(const Resource& resource) override;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

template <class Rbf>
using FmmSymmetricEvaluator = FmmGenericSymmetricEvaluator<Kernel<Rbf>>;

template <class Rbf>
using FmmHessianSymmetricEvaluator = FmmGenericSymmetricEvaluator<HessianKernel<Rbf>>;

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim, 1> make_fmm_symmetric_evaluator(
    const rbf::Rbf<Dim>& rbf, const geometry::Bbox<Dim>& bbox);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim, Dim> make_fmm_hessian_symmetric_evaluator(
    const rbf::Rbf<Dim>& rbf, const geometry::Bbox<Dim>& bbox);

}  // namespace polatory::fmm
