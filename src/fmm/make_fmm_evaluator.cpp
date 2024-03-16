#include <polatory/fmm/fmm_evaluator.hpp>
#include <polatory/fmm/fmm_symmetric_evaluator.hpp>
#include <polatory/rbf/cov_cauchy3.hpp>
#include <polatory/rbf/cov_cauchy5.hpp>
#include <polatory/rbf/cov_cauchy7.hpp>
#include <polatory/rbf/cov_cauchy9.hpp>
#include <polatory/rbf/cov_cubic.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_gaussian.hpp>
#include <polatory/rbf/cov_spherical.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
#include <polatory/rbf/multiquadric.hpp>
#include <polatory/rbf/polyharmonic_even.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <stdexcept>

using polatory::geometry::bboxNd;
using polatory::rbf::rbf_proxy;
using polatory::rbf::internal::biharmonic2d;
using polatory::rbf::internal::biharmonic3d;
using polatory::rbf::internal::cov_cauchy3;
using polatory::rbf::internal::cov_cauchy5;
using polatory::rbf::internal::cov_cauchy7;
using polatory::rbf::internal::cov_cauchy9;
using polatory::rbf::internal::cov_cubic;
using polatory::rbf::internal::cov_exponential;
using polatory::rbf::internal::cov_gaussian;
using polatory::rbf::internal::cov_spherical;
using polatory::rbf::internal::cov_spheroidal3;
using polatory::rbf::internal::cov_spheroidal5;
using polatory::rbf::internal::cov_spheroidal7;
using polatory::rbf::internal::cov_spheroidal9;
using polatory::rbf::internal::inverse_multiquadric1;
using polatory::rbf::internal::multiquadric1;
using polatory::rbf::internal::multiquadric3;
using polatory::rbf::internal::triharmonic2d;
using polatory::rbf::internal::triharmonic3d;

namespace polatory::fmm {

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_evaluator(const rbf_proxy<Dim>& rbf, const bboxNd<Dim>& bbox,
                                               int order) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                            \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                       \
    return std::make_unique<fmm_evaluator<RBF_NAME<Dim>>>(*derived, bbox, order); \
  }

  CASE(biharmonic2d);
  CASE(biharmonic3d);
  CASE(cov_cauchy3);
  CASE(cov_cauchy5);
  CASE(cov_cauchy7);
  CASE(cov_cauchy9);
  CASE(cov_cubic);
  CASE(cov_exponential);
  CASE(cov_gaussian);
  CASE(cov_spherical);
  CASE(cov_spheroidal3);
  CASE(cov_spheroidal5);
  CASE(cov_spheroidal7);
  CASE(cov_spheroidal9);
  CASE(inverse_multiquadric1);
  CASE(multiquadric1);
  CASE(multiquadric3);
  CASE(triharmonic2d);
  CASE(triharmonic3d);

#undef CASE

  throw std::invalid_argument("RBF type not supported.");
}

template FmmGenericEvaluatorPtr<1> make_fmm_evaluator<1>(const rbf_proxy<1>& rbf,
                                                         const bboxNd<1>& bbox, int order);

template FmmGenericEvaluatorPtr<2> make_fmm_evaluator<2>(const rbf_proxy<2>& rbf,
                                                         const bboxNd<2>& bbox, int order);

template FmmGenericEvaluatorPtr<3> make_fmm_evaluator<3>(const rbf_proxy<3>& rbf,
                                                         const bboxNd<3>& bbox, int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_evaluator(const rbf_proxy<Dim>& rbf,
                                                        const bboxNd<Dim>& bbox, int order) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                                     \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                \
    return std::make_unique<fmm_gradient_evaluator<RBF_NAME<Dim>>>(*derived, bbox, order); \
  }

  CASE(biharmonic2d);
  CASE(biharmonic3d);
  CASE(cov_cauchy3);
  CASE(cov_cauchy5);
  CASE(cov_cauchy7);
  CASE(cov_cauchy9);
  CASE(cov_cubic);
  CASE(cov_exponential);
  CASE(cov_gaussian);
  CASE(cov_spherical);
  CASE(cov_spheroidal3);
  CASE(cov_spheroidal5);
  CASE(cov_spheroidal7);
  CASE(cov_spheroidal9);
  CASE(inverse_multiquadric1);
  CASE(multiquadric1);
  CASE(multiquadric3);
  CASE(triharmonic2d);
  CASE(triharmonic3d);

#undef CASE

  throw std::invalid_argument("RBF type not supported.");
}

template FmmGenericEvaluatorPtr<1> make_fmm_gradient_evaluator<1>(const rbf_proxy<1>& rbf,
                                                                  const bboxNd<1>& bbox, int order);

template FmmGenericEvaluatorPtr<2> make_fmm_gradient_evaluator<2>(const rbf_proxy<2>& rbf,
                                                                  const bboxNd<2>& bbox, int order);

template FmmGenericEvaluatorPtr<3> make_fmm_gradient_evaluator<3>(const rbf_proxy<3>& rbf,
                                                                  const bboxNd<3>& bbox, int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_transpose_evaluator(const rbf_proxy<Dim>& rbf,
                                                                  const bboxNd<Dim>& bbox,
                                                                  int order) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                                       \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                  \
    return std::make_unique<fmm_gradient_transpose_evaluator<RBF_NAME<Dim>>>(*derived, bbox, \
                                                                             order);         \
  }

  CASE(biharmonic2d);
  CASE(biharmonic3d);
  CASE(cov_cauchy3);
  CASE(cov_cauchy5);
  CASE(cov_cauchy7);
  CASE(cov_cauchy9);
  CASE(cov_cubic);
  CASE(cov_exponential);
  CASE(cov_gaussian);
  CASE(cov_spherical);
  CASE(cov_spheroidal3);
  CASE(cov_spheroidal5);
  CASE(cov_spheroidal7);
  CASE(cov_spheroidal9);
  CASE(inverse_multiquadric1);
  CASE(multiquadric1);
  CASE(multiquadric3);
  CASE(triharmonic2d);
  CASE(triharmonic3d);

#undef CASE

  throw std::invalid_argument("RBF type not supported.");
}

template FmmGenericEvaluatorPtr<1> make_fmm_gradient_transpose_evaluator<1>(const rbf_proxy<1>& rbf,
                                                                            const bboxNd<1>& bbox,
                                                                            int order);

template FmmGenericEvaluatorPtr<2> make_fmm_gradient_transpose_evaluator<2>(const rbf_proxy<2>& rbf,
                                                                            const bboxNd<2>& bbox,
                                                                            int order);

template FmmGenericEvaluatorPtr<3> make_fmm_gradient_transpose_evaluator<3>(const rbf_proxy<3>& rbf,
                                                                            const bboxNd<3>& bbox,
                                                                            int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_hessian_evaluator(const rbf_proxy<Dim>& rbf,
                                                       const bboxNd<Dim>& bbox, int order) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                                    \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                               \
    return std::make_unique<fmm_hessian_evaluator<RBF_NAME<Dim>>>(*derived, bbox, order); \
  }

  CASE(biharmonic2d);
  CASE(biharmonic3d);
  CASE(cov_cauchy3);
  CASE(cov_cauchy5);
  CASE(cov_cauchy7);
  CASE(cov_cauchy9);
  CASE(cov_cubic);
  CASE(cov_exponential);
  CASE(cov_gaussian);
  CASE(cov_spherical);
  CASE(cov_spheroidal3);
  CASE(cov_spheroidal5);
  CASE(cov_spheroidal7);
  CASE(cov_spheroidal9);
  CASE(inverse_multiquadric1);
  CASE(multiquadric1);
  CASE(multiquadric3);
  CASE(triharmonic2d);
  CASE(triharmonic3d);

#undef CASE

  throw std::invalid_argument("RBF type not supported.");
}

template FmmGenericEvaluatorPtr<1> make_fmm_hessian_evaluator<1>(const rbf_proxy<1>& rbf,
                                                                 const bboxNd<1>& bbox, int order);

template FmmGenericEvaluatorPtr<2> make_fmm_hessian_evaluator<2>(const rbf_proxy<2>& rbf,
                                                                 const bboxNd<2>& bbox, int order);

template FmmGenericEvaluatorPtr<3> make_fmm_hessian_evaluator<3>(const rbf_proxy<3>& rbf,
                                                                 const bboxNd<3>& bbox, int order);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_symmetric_evaluator(const rbf_proxy<Dim>& rbf,
                                                                  const bboxNd<Dim>& bbox,
                                                                  int order) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                                      \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                 \
    return std::make_unique<fmm_symmetric_evaluator<RBF_NAME<Dim>>>(*derived, bbox, order); \
  }

  CASE(biharmonic2d);
  CASE(biharmonic3d);
  CASE(cov_cauchy3);
  CASE(cov_cauchy5);
  CASE(cov_cauchy7);
  CASE(cov_cauchy9);
  CASE(cov_cubic);
  CASE(cov_exponential);
  CASE(cov_gaussian);
  CASE(cov_spherical);
  CASE(cov_spheroidal3);
  CASE(cov_spheroidal5);
  CASE(cov_spheroidal7);
  CASE(cov_spheroidal9);
  CASE(inverse_multiquadric1);
  CASE(multiquadric1);
  CASE(multiquadric3);
  CASE(triharmonic2d);
  CASE(triharmonic3d);

#undef CASE

  throw std::runtime_error("RBF type not supported.");
}

template FmmGenericSymmetricEvaluatorPtr<1> make_fmm_symmetric_evaluator<1>(const rbf_proxy<1>& rbf,
                                                                            const bboxNd<1>& bbox,
                                                                            int order);

template FmmGenericSymmetricEvaluatorPtr<2> make_fmm_symmetric_evaluator<2>(const rbf_proxy<2>& rbf,
                                                                            const bboxNd<2>& bbox,
                                                                            int order);

template FmmGenericSymmetricEvaluatorPtr<3> make_fmm_symmetric_evaluator<3>(const rbf_proxy<3>& rbf,
                                                                            const bboxNd<3>& bbox,
                                                                            int order);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_hessian_symmetric_evaluator(const rbf_proxy<Dim>& rbf,
                                                                          const bboxNd<Dim>& bbox,
                                                                          int order) {
  auto* base = rbf.get_raw_pointer();

#define CASE(RBF_NAME)                                                                      \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                 \
    return std::make_unique<fmm_hessian_symmetric_evaluator<RBF_NAME<Dim>>>(*derived, bbox, \
                                                                            order);         \
  }

  CASE(biharmonic2d);
  CASE(biharmonic3d);
  CASE(cov_cauchy3);
  CASE(cov_cauchy5);
  CASE(cov_cauchy7);
  CASE(cov_cauchy9);
  CASE(cov_cubic);
  CASE(cov_exponential);
  CASE(cov_gaussian);
  CASE(cov_spherical);
  CASE(cov_spheroidal3);
  CASE(cov_spheroidal5);
  CASE(cov_spheroidal7);
  CASE(cov_spheroidal9);
  CASE(inverse_multiquadric1);
  CASE(multiquadric1);
  CASE(multiquadric3);
  CASE(triharmonic2d);
  CASE(triharmonic3d);

#undef CASE

  throw std::runtime_error("RBF type not supported.");
}

template FmmGenericSymmetricEvaluatorPtr<1> make_fmm_hessian_symmetric_evaluator<1>(
    const rbf_proxy<1>& rbf, const bboxNd<1>& bbox, int order);

template FmmGenericSymmetricEvaluatorPtr<2> make_fmm_hessian_symmetric_evaluator<2>(
    const rbf_proxy<2>& rbf, const bboxNd<2>& bbox, int order);

template FmmGenericSymmetricEvaluatorPtr<3> make_fmm_hessian_symmetric_evaluator<3>(
    const rbf_proxy<3>& rbf, const bboxNd<3>& bbox, int order);

}  // namespace polatory::fmm
