#pragma once

#include <Eigen/Core>
#include <memory>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/geometry/bbox3d.hpp>
#include <polatory/geometry/point3d.hpp>
#include <polatory/rbf/rbf_base.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <polatory/types.hpp>

namespace polatory::fmm {

template <int Dim>
class fmm_generic_symmetric_evaluator_base {
  static constexpr int kDim = Dim;
  using Points = geometry::pointsNd<kDim>;

 public:
  fmm_generic_symmetric_evaluator_base() = default;

  virtual ~fmm_generic_symmetric_evaluator_base() = default;

  fmm_generic_symmetric_evaluator_base(const fmm_generic_symmetric_evaluator_base&) = delete;
  fmm_generic_symmetric_evaluator_base(fmm_generic_symmetric_evaluator_base&&) = delete;
  fmm_generic_symmetric_evaluator_base& operator=(const fmm_generic_symmetric_evaluator_base&) =
      delete;
  fmm_generic_symmetric_evaluator_base& operator=(fmm_generic_symmetric_evaluator_base&&) = delete;

  virtual vectord evaluate() const = 0;

  virtual void set_points(const Points& points) = 0;

  virtual void set_weights(const Eigen::Ref<const vectord>& weights) = 0;
};

template <int Dim>
using FmmGenericSymmetricEvaluatorPtr = std::unique_ptr<fmm_generic_symmetric_evaluator_base<Dim>>;

template <class Rbf, class Kernel>
class fmm_generic_symmetric_evaluator final
    : public fmm_generic_symmetric_evaluator_base<Rbf::kDim> {
  static constexpr int kDim = Rbf::kDim;
  using Bbox = geometry::bboxNd<kDim>;
  using Points = geometry::pointsNd<kDim>;

 public:
  fmm_generic_symmetric_evaluator(const Rbf& rbf, const Bbox& bbox, double accuracy);

  ~fmm_generic_symmetric_evaluator() override;

  fmm_generic_symmetric_evaluator(const fmm_generic_symmetric_evaluator&) = delete;
  fmm_generic_symmetric_evaluator(fmm_generic_symmetric_evaluator&&) = delete;
  fmm_generic_symmetric_evaluator& operator=(const fmm_generic_symmetric_evaluator&) = delete;
  fmm_generic_symmetric_evaluator& operator=(fmm_generic_symmetric_evaluator&&) = delete;

  vectord evaluate() const override;

  void set_points(const Points& points) override;

  void set_weights(const Eigen::Ref<const vectord>& weights) override;

 private:
  class impl;

  std::unique_ptr<impl> impl_;
};

template <class Rbf>
using fmm_symmetric_evaluator = fmm_generic_symmetric_evaluator<Rbf, kernel<Rbf>>;

template <class Rbf>
using fmm_hessian_symmetric_evaluator = fmm_generic_symmetric_evaluator<Rbf, hessian_kernel<Rbf>>;

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_symmetric_evaluator(const rbf::rbf_proxy<Dim>& rbf,
                                                                  const geometry::bboxNd<Dim>& bbox,
                                                                  double accuracy);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_hessian_symmetric_evaluator(
    const rbf::rbf_proxy<Dim>& rbf, const geometry::bboxNd<Dim>& bbox, double accuracy);

}  // namespace polatory::fmm
