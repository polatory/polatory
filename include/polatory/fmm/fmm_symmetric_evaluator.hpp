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

template <int Dim>
class FmmGenericSymmetricEvaluatorBase {
  static constexpr int kDim = Dim;
  using Points = geometry::Points<kDim>;

 public:
  FmmGenericSymmetricEvaluatorBase() = default;

  virtual ~FmmGenericSymmetricEvaluatorBase() = default;

  FmmGenericSymmetricEvaluatorBase(const FmmGenericSymmetricEvaluatorBase&) = delete;
  FmmGenericSymmetricEvaluatorBase(FmmGenericSymmetricEvaluatorBase&&) = delete;
  FmmGenericSymmetricEvaluatorBase& operator=(const FmmGenericSymmetricEvaluatorBase&) = delete;
  FmmGenericSymmetricEvaluatorBase& operator=(FmmGenericSymmetricEvaluatorBase&&) = delete;

  virtual VecX evaluate() const = 0;

  virtual void set_accuracy(double accuracy) = 0;

  virtual void set_points(const Points& points) = 0;

  virtual void set_weights(const Eigen::Ref<const VecX>& weights) = 0;
};

template <int Dim>
using FmmGenericSymmetricEvaluatorPtr = std::unique_ptr<FmmGenericSymmetricEvaluatorBase<Dim>>;

template <class Kernel>
class FmmGenericSymmetricEvaluator final : public FmmGenericSymmetricEvaluatorBase<Kernel::kDim> {
  using Rbf = typename Kernel::Rbf;
  static constexpr int kDim = Kernel::kDim;

  using Bbox = geometry::Bbox<kDim>;
  using Points = geometry::Points<kDim>;

 public:
  FmmGenericSymmetricEvaluator(const Rbf& rbf, const Bbox& bbox);

  ~FmmGenericSymmetricEvaluator() override;

  FmmGenericSymmetricEvaluator(const FmmGenericSymmetricEvaluator&) = delete;
  FmmGenericSymmetricEvaluator(FmmGenericSymmetricEvaluator&&) = delete;
  FmmGenericSymmetricEvaluator& operator=(const FmmGenericSymmetricEvaluator&) = delete;
  FmmGenericSymmetricEvaluator& operator=(FmmGenericSymmetricEvaluator&&) = delete;

  VecX evaluate() const override;

  void set_accuracy(double accuracy) override;

  void set_points(const Points& points) override;

  void set_weights(const Eigen::Ref<const VecX>& weights) override;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

template <class Rbf>
using FmmSymmetricEvaluator = FmmGenericSymmetricEvaluator<Kernel<Rbf>>;

template <class Rbf>
using FmmHessianSymmetricEvaluator = FmmGenericSymmetricEvaluator<HessianKernel<Rbf>>;

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_symmetric_evaluator(const rbf::Rbf<Dim>& rbf,
                                                                  const geometry::Bbox<Dim>& bbox);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_hessian_symmetric_evaluator(
    const rbf::Rbf<Dim>& rbf, const geometry::Bbox<Dim>& bbox);

}  // namespace polatory::fmm
