#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/fmm/fmm_evaluator.hpp>

namespace polatory::fmm {

template <class Rbf, class Kernel>
class FmmGenericEvaluator<Rbf, Kernel>::Impl {
  using RbfDirectPart = typename Rbf::DirectPart;
  using RbfFastPart = typename Rbf::FastPart;
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

  void set_weights(const Eigen::Ref<const VecX>& weights) {
    POLATORY_ASSERT(weights.rows() == km * n_src_points_);
    direct_eval_.set_weights(weights);
    fast_eval_.set_weights(weights);
  }

 private:
  RbfDirectPart rbf_direct_part_;
  RbfFastPart rbf_fast_part_;
  FmmGenericEvaluator<RbfDirectPart, Kernel> direct_eval_;
  FmmGenericEvaluator<RbfFastPart, Kernel> fast_eval_;

  Index n_src_points_{};
  Index n_trg_points_{};
};

template <class Rbf, class Kernel>
FmmGenericEvaluator<Rbf, Kernel>::FmmGenericEvaluator(const Rbf& rbf, const Bbox& bbox)
    : impl_(std::make_unique<Impl>(rbf, bbox)) {}

template <class Rbf, class Kernel>
FmmGenericEvaluator<Rbf, Kernel>::~FmmGenericEvaluator() = default;

template <class Rbf, class Kernel>
VecX FmmGenericEvaluator<Rbf, Kernel>::evaluate() const {
  return impl_->evaluate();
}

template <class Rbf, class Kernel>
void FmmGenericEvaluator<Rbf, Kernel>::set_accuracy(double accuracy) {
  impl_->set_accuracy(accuracy);
}

template <class Rbf, class Kernel>
void FmmGenericEvaluator<Rbf, Kernel>::set_source_points(const Points& points) {
  impl_->set_source_points(points);
}

template <class Rbf, class Kernel>
void FmmGenericEvaluator<Rbf, Kernel>::set_target_points(const Points& points) {
  impl_->set_target_points(points);
}

template <class Rbf, class Kernel>
void FmmGenericEvaluator<Rbf, Kernel>::set_weights(const Eigen::Ref<const VecX>& weights) {
  impl_->set_weights(weights);
}

#define IMPLEMENT_FMM_EVALUATORS_(RBF)                                   \
  template class FmmGenericEvaluator<RBF, Kernel<RBF>>;                  \
  template class FmmGenericEvaluator<RBF, GradientKernel<RBF>>;          \
  template class FmmGenericEvaluator<RBF, GradientTransposeKernel<RBF>>; \
  template class FmmGenericEvaluator<RBF, HessianKernel<RBF>>;

#define IMPLEMENT_FMM_EVALUATORS(RBF_NAME) \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<1>);  \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<2>);  \
  IMPLEMENT_FMM_EVALUATORS_(RBF_NAME<3>);

}  // namespace polatory::fmm
