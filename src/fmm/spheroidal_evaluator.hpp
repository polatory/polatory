#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>

namespace polatory::fmm {

template <class Kernel>
class FmmGenericEvaluator<Kernel>::Impl {
  using RbfDirectPart = typename Rbf::DirectPart;
  using RbfFastPart = typename Rbf::FastPart;
  using KernelDirectPart = typename Kernel::template Rebind<RbfDirectPart>;
  using KernelFastPart = typename Kernel::template Rebind<RbfFastPart>;
  static constexpr int km{Kernel::km};
  static constexpr int kn{Kernel::kn};

 public:
  Impl(const Rbf& rbf, const Bbox& bbox)
      : rbf_direct_part_{rbf.direct_part()},
        rbf_fast_part_{rbf.fast_part()},
        direct_eval_{rbf_direct_part_, bbox},
        fast_eval_{rbf_fast_part_, bbox} {}

  VecX evaluate() const {
    VecX y = VecX::Zero(kn * n_trg_points_);
    y += direct_eval_.evaluate();
    y += fast_eval_.evaluate();
    return y;
  }

  void set_accuracy(double accuracy) {
    direct_eval_.set_accuracy(accuracy);
    fast_eval_.set_accuracy(accuracy);
  }

  void set_source_points(const Points& points) {
    n_src_points_ = points.rows();
    direct_eval_.set_source_points(points);
    fast_eval_.set_source_points(points);
  }

  void set_target_points(const Points& points) {
    n_trg_points_ = points.rows();
    direct_eval_.set_target_points(points);
    fast_eval_.set_target_points(points);
  }

 private:
  RbfDirectPart rbf_direct_part_;
  RbfFastPart rbf_fast_part_;
  FmmGenericEvaluator<KernelDirectPart> direct_eval_;
  FmmGenericEvaluator<KernelFastPart> fast_eval_;

  Index n_src_points_{};
  Index n_trg_points_{};
};

template <class Kernel>
FmmGenericEvaluator<Kernel>::FmmGenericEvaluator(const Rbf& rbf, const Bbox& bbox)
    : impl_(std::make_unique<Impl>(rbf, bbox)) {}

template <class Kernel>
FmmGenericEvaluator<Kernel>::~FmmGenericEvaluator() = default;

template <class Kernel>
VecX FmmGenericEvaluator<Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Kernel>
void FmmGenericEvaluator<Kernel>::set_accuracy(double accuracy) {
  impl_->set_accuracy(accuracy);
}

template <class Kernel>
void FmmGenericEvaluator<Kernel>::set_source_resource(const Resource& resource) {
  impl_->set_source_resource(resource);
}

template <class Kernel>
void FmmGenericEvaluator<Kernel>::set_target_resource(const Resource& resource) {
  impl_->set_target_resource(resource);
}

#define IMPLEMENT_FMM_EVALUATORS_(RBF)                              \
  template class FmmGenericEvaluator<Kernel<RBF>>;                  \
  template class FmmGenericEvaluator<GradientKernel<RBF>>;          \
  template class FmmGenericEvaluator<GradientTransposeKernel<RBF>>; \
  template class FmmGenericEvaluator<HessianKernel<RBF>>;

#define IMPLEMENT_FMM_EVALUATORS(RBF_NAME) \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<1>);  \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<2>);  \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<3>);

#define EXTERN_FMM_EVALUATORS_(RBF)                                        \
  extern template class FmmGenericEvaluator<Kernel<RBF>>;                  \
  extern template class FmmGenericEvaluator<GradientKernel<RBF>>;          \
  extern template class FmmGenericEvaluator<GradientTransposeKernel<RBF>>; \
  extern template class FmmGenericEvaluator<HessianKernel<RBF>>;

#define EXTERN_FMM_EVALUATORS(RBF_NAME) \
  EXTERN_FMM_EVALUATORS_(RBF_NAME<1>);  \
  EXTERN_FMM_EVALUATORS_(RBF_NAME<2>);  \
  EXTERN_FMM_EVALUATORS_(RBF_NAME<3>);

}  // namespace polatory::fmm
