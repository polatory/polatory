#include <jizai/fmm/fmm_evaluator.hpp>
#include <jizai/fmm/fmm_symmetric_evaluator.hpp>
#include <jizai/rbf/cov_cubic.hpp>
#include <jizai/rbf/cov_exponential.hpp>
#include <jizai/rbf/cov_gaussian.hpp>
#include <jizai/rbf/cov_generalized_cauchy3.hpp>
#include <jizai/rbf/cov_generalized_cauchy5.hpp>
#include <jizai/rbf/cov_generalized_cauchy7.hpp>
#include <jizai/rbf/cov_generalized_cauchy9.hpp>
#include <jizai/rbf/cov_spherical.hpp>
#include <jizai/rbf/cov_spheroidal3.hpp>
#include <jizai/rbf/cov_spheroidal5.hpp>
#include <jizai/rbf/cov_spheroidal7.hpp>
#include <jizai/rbf/cov_spheroidal9.hpp>
#include <jizai/rbf/polyharmonic_even.hpp>
#include <jizai/rbf/polyharmonic_odd.hpp>
#include <stdexcept>

using jizai::geometry::Bbox;
using jizai::rbf::Rbf;
using jizai::rbf::internal::Biharmonic2D;
using jizai::rbf::internal::Biharmonic3D;
using jizai::rbf::internal::CovCubic;
using jizai::rbf::internal::CovExponential;
using jizai::rbf::internal::CovGaussian;
using jizai::rbf::internal::CovGeneralizedCauchy3;
using jizai::rbf::internal::CovGeneralizedCauchy5;
using jizai::rbf::internal::CovGeneralizedCauchy7;
using jizai::rbf::internal::CovGeneralizedCauchy9;
using jizai::rbf::internal::CovSpherical;
using jizai::rbf::internal::CovSpheroidal3;
using jizai::rbf::internal::CovSpheroidal5;
using jizai::rbf::internal::CovSpheroidal7;
using jizai::rbf::internal::CovSpheroidal9;
using jizai::rbf::internal::Triharmonic2D;
using jizai::rbf::internal::Triharmonic3D;

namespace jizai::fmm {

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_evaluator(const Rbf<Dim>& rbf, const Bbox<Dim>& bbox) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                    \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {               \
    return std::make_unique<FmmEvaluator<RBF_NAME<Dim>>>(*derived, bbox); \
  }

  CASE(Biharmonic2D);
  CASE(Biharmonic3D);
  CASE(CovCubic);
  CASE(CovExponential);
  CASE(CovGaussian);
  CASE(CovGeneralizedCauchy3);
  CASE(CovGeneralizedCauchy5);
  CASE(CovGeneralizedCauchy7);
  CASE(CovGeneralizedCauchy9);
  CASE(CovSpherical);
  CASE(CovSpheroidal3);
  CASE(CovSpheroidal5);
  CASE(CovSpheroidal7);
  CASE(CovSpheroidal9);
  CASE(Triharmonic2D);
  CASE(Triharmonic3D);

#undef CASE

  throw std::runtime_error("not implemented");
}

template FmmGenericEvaluatorPtr<1> make_fmm_evaluator<1>(const Rbf<1>& rbf, const Bbox<1>& bbox);

template FmmGenericEvaluatorPtr<2> make_fmm_evaluator<2>(const Rbf<2>& rbf, const Bbox<2>& bbox);

template FmmGenericEvaluatorPtr<3> make_fmm_evaluator<3>(const Rbf<3>& rbf, const Bbox<3>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_evaluator(const Rbf<Dim>& rbf,
                                                        const Bbox<Dim>& bbox) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                            \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                       \
    return std::make_unique<FmmGradientEvaluator<RBF_NAME<Dim>>>(*derived, bbox); \
  }

  CASE(Biharmonic2D);
  CASE(Biharmonic3D);
  CASE(CovCubic);
  CASE(CovExponential);
  CASE(CovGaussian);
  CASE(CovGeneralizedCauchy3);
  CASE(CovGeneralizedCauchy5);
  CASE(CovGeneralizedCauchy7);
  CASE(CovGeneralizedCauchy9);
  CASE(CovSpherical);
  CASE(CovSpheroidal3);
  CASE(CovSpheroidal5);
  CASE(CovSpheroidal7);
  CASE(CovSpheroidal9);
  CASE(Triharmonic2D);
  CASE(Triharmonic3D);

#undef CASE

  throw std::runtime_error("not implemented");
}

template FmmGenericEvaluatorPtr<1> make_fmm_gradient_evaluator<1>(const Rbf<1>& rbf,
                                                                  const Bbox<1>& bbox);

template FmmGenericEvaluatorPtr<2> make_fmm_gradient_evaluator<2>(const Rbf<2>& rbf,
                                                                  const Bbox<2>& bbox);

template FmmGenericEvaluatorPtr<3> make_fmm_gradient_evaluator<3>(const Rbf<3>& rbf,
                                                                  const Bbox<3>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_transpose_evaluator(const Rbf<Dim>& rbf,
                                                                  const Bbox<Dim>& bbox) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                                     \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                \
    return std::make_unique<FmmGradientTransposeEvaluator<RBF_NAME<Dim>>>(*derived, bbox); \
  }

  CASE(Biharmonic2D);
  CASE(Biharmonic3D);
  CASE(CovCubic);
  CASE(CovExponential);
  CASE(CovGaussian);
  CASE(CovGeneralizedCauchy3);
  CASE(CovGeneralizedCauchy5);
  CASE(CovGeneralizedCauchy7);
  CASE(CovGeneralizedCauchy9);
  CASE(CovSpherical);
  CASE(CovSpheroidal3);
  CASE(CovSpheroidal5);
  CASE(CovSpheroidal7);
  CASE(CovSpheroidal9);
  CASE(Triharmonic2D);
  CASE(Triharmonic3D);

#undef CASE

  throw std::runtime_error("not implemented");
}

template FmmGenericEvaluatorPtr<1> make_fmm_gradient_transpose_evaluator<1>(const Rbf<1>& rbf,
                                                                            const Bbox<1>& bbox);

template FmmGenericEvaluatorPtr<2> make_fmm_gradient_transpose_evaluator<2>(const Rbf<2>& rbf,
                                                                            const Bbox<2>& bbox);

template FmmGenericEvaluatorPtr<3> make_fmm_gradient_transpose_evaluator<3>(const Rbf<3>& rbf,
                                                                            const Bbox<3>& bbox);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_hessian_evaluator(const Rbf<Dim>& rbf, const Bbox<Dim>& bbox) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                           \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                      \
    return std::make_unique<FmmHessianEvaluator<RBF_NAME<Dim>>>(*derived, bbox); \
  }

  CASE(Biharmonic2D);
  CASE(Biharmonic3D);
  CASE(CovCubic);
  CASE(CovExponential);
  CASE(CovGaussian);
  CASE(CovGeneralizedCauchy3);
  CASE(CovGeneralizedCauchy5);
  CASE(CovGeneralizedCauchy7);
  CASE(CovGeneralizedCauchy9);
  CASE(CovSpherical);
  CASE(CovSpheroidal3);
  CASE(CovSpheroidal5);
  CASE(CovSpheroidal7);
  CASE(CovSpheroidal9);
  CASE(Triharmonic2D);
  CASE(Triharmonic3D);

#undef CASE

  throw std::runtime_error("not implemented");
}

template FmmGenericEvaluatorPtr<1> make_fmm_hessian_evaluator<1>(const Rbf<1>& rbf,
                                                                 const Bbox<1>& bbox);

template FmmGenericEvaluatorPtr<2> make_fmm_hessian_evaluator<2>(const Rbf<2>& rbf,
                                                                 const Bbox<2>& bbox);

template FmmGenericEvaluatorPtr<3> make_fmm_hessian_evaluator<3>(const Rbf<3>& rbf,
                                                                 const Bbox<3>& bbox);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_symmetric_evaluator(const Rbf<Dim>& rbf,
                                                                  const Bbox<Dim>& bbox) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                             \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                        \
    return std::make_unique<FmmSymmetricEvaluator<RBF_NAME<Dim>>>(*derived, bbox); \
  }

  CASE(Biharmonic2D);
  CASE(Biharmonic3D);
  CASE(CovCubic);
  CASE(CovExponential);
  CASE(CovGaussian);
  CASE(CovGeneralizedCauchy3);
  CASE(CovGeneralizedCauchy5);
  CASE(CovGeneralizedCauchy7);
  CASE(CovGeneralizedCauchy9);
  CASE(CovSpherical);
  CASE(CovSpheroidal3);
  CASE(CovSpheroidal5);
  CASE(CovSpheroidal7);
  CASE(CovSpheroidal9);
  CASE(Triharmonic2D);
  CASE(Triharmonic3D);

#undef CASE

  throw std::runtime_error("not implemented");
}

template FmmGenericSymmetricEvaluatorPtr<1> make_fmm_symmetric_evaluator<1>(const Rbf<1>& rbf,
                                                                            const Bbox<1>& bbox);

template FmmGenericSymmetricEvaluatorPtr<2> make_fmm_symmetric_evaluator<2>(const Rbf<2>& rbf,
                                                                            const Bbox<2>& bbox);

template FmmGenericSymmetricEvaluatorPtr<3> make_fmm_symmetric_evaluator<3>(const Rbf<3>& rbf,
                                                                            const Bbox<3>& bbox);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_hessian_symmetric_evaluator(const Rbf<Dim>& rbf,
                                                                          const Bbox<Dim>& bbox) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                                    \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                               \
    return std::make_unique<FmmHessianSymmetricEvaluator<RBF_NAME<Dim>>>(*derived, bbox); \
  }

  CASE(Biharmonic2D);
  CASE(Biharmonic3D);
  CASE(CovCubic);
  CASE(CovExponential);
  CASE(CovGaussian);
  CASE(CovGeneralizedCauchy3);
  CASE(CovGeneralizedCauchy5);
  CASE(CovGeneralizedCauchy7);
  CASE(CovGeneralizedCauchy9);
  CASE(CovSpherical);
  CASE(CovSpheroidal3);
  CASE(CovSpheroidal5);
  CASE(CovSpheroidal7);
  CASE(CovSpheroidal9);
  CASE(Triharmonic2D);
  CASE(Triharmonic3D);

#undef CASE

  throw std::runtime_error("not implemented");
}

template FmmGenericSymmetricEvaluatorPtr<1> make_fmm_hessian_symmetric_evaluator<1>(
    const Rbf<1>& rbf, const Bbox<1>& bbox);

template FmmGenericSymmetricEvaluatorPtr<2> make_fmm_hessian_symmetric_evaluator<2>(
    const Rbf<2>& rbf, const Bbox<2>& bbox);

template FmmGenericSymmetricEvaluatorPtr<3> make_fmm_hessian_symmetric_evaluator<3>(
    const Rbf<3>& rbf, const Bbox<3>& bbox);

}  // namespace jizai::fmm
