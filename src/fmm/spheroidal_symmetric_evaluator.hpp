#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>

namespace polatory::fmm {

template <class Kernel>
class FmmGenericSymmetricEvaluator<Kernel>::Impl {
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
    VecX y = VecX::Zero(kn * n_points_);
    y += direct_eval_.evaluate();
    y += fast_eval_.evaluate();
    return y;
  }

  void set_accuracy(double accuracy) {
    direct_eval_.set_accuracy(accuracy);
    fast_eval_.set_accuracy(accuracy);
  }

  void set_points(const Points& points) {
    n_points_ = points.rows();
    direct_eval_.set_points(points);
    fast_eval_.set_points(points);
  }

  void set_weights(const Eigen::Ref<const VecX>& weights) {
    POLATORY_ASSERT(weights.rows() == km * n_points_);
    direct_eval_.set_weights(weights);
    fast_eval_.set_weights(weights);
  }

 private:
  RbfDirectPart rbf_direct_part_;
  RbfFastPart rbf_fast_part_;
  FmmGenericSymmetricEvaluator<KernelDirectPart> direct_eval_;
  FmmGenericSymmetricEvaluator<KernelFastPart> fast_eval_;

  Index n_points_{};
};

template <class Kernel>
FmmGenericSymmetricEvaluator<Kernel>::FmmGenericSymmetricEvaluator(const Rbf& rbf, const Bbox& bbox)
    : impl_(std::make_unique<Impl>(rbf, bbox)) {}

template <class Kernel>
FmmGenericSymmetricEvaluator<Kernel>::~FmmGenericSymmetricEvaluator() = default;

template <class Kernel>
VecX FmmGenericSymmetricEvaluator<Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Kernel>
void FmmGenericSymmetricEvaluator<Kernel>::set_accuracy(double accuracy) {
  impl_->set_accuracy(accuracy);
}

template <class Kernel>
void FmmGenericSymmetricEvaluator<Kernel>::set_points(const Points& points) {
  impl_->set_points(points);
}

template <class Kernel>
void FmmGenericSymmetricEvaluator<Kernel>::set_weights(const Eigen::Ref<const VecX>& weights) {
  impl_->set_weights(weights);
}

#define IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF)            \
  template class FmmGenericSymmetricEvaluator<Kernel<RBF>>; \
  template class FmmGenericSymmetricEvaluator<HessianKernel<RBF>>;

#define IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(RBF_NAME) \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<1>);  \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<2>);  \
  IMPLEMENT_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<3>);

#define EXTERN_FMM_SYMMETRIC_EVALUATORS_(RBF)                      \
  extern template class FmmGenericSymmetricEvaluator<Kernel<RBF>>; \
  extern template class FmmGenericSymmetricEvaluator<HessianKernel<RBF>>;

#define EXTERN_FMM_SYMMETRIC_EVALUATORS(RBF_NAME) \
  EXTERN_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<1>);  \
  EXTERN_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<2>);  \
  EXTERN_FMM_SYMMETRIC_EVALUATORS_(RBF_NAME<3>);

}  // namespace polatory::fmm
