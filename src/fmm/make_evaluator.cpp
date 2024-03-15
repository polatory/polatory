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
using polatory::rbf::biharmonic2d;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_cauchy3;
using polatory::rbf::cov_cauchy5;
using polatory::rbf::cov_cauchy7;
using polatory::rbf::cov_cauchy9;
using polatory::rbf::cov_cubic;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_gaussian;
using polatory::rbf::cov_spherical;
using polatory::rbf::cov_spheroidal3;
using polatory::rbf::cov_spheroidal5;
using polatory::rbf::cov_spheroidal7;
using polatory::rbf::cov_spheroidal9;
using polatory::rbf::inverse_multiquadric1;
using polatory::rbf::multiquadric1;
using polatory::rbf::multiquadric3;
using polatory::rbf::RbfPtr;
using polatory::rbf::triharmonic2d;
using polatory::rbf::triharmonic3d;

namespace polatory::fmm {

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_evaluator(const RbfPtr<Dim>& rbf, const bboxNd<Dim>& bbox,
                                               int order) {
  auto* base = rbf.get();

#define IMPL(RBF_NAME)                                                            \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                       \
    return std::make_unique<fmm_evaluator<RBF_NAME<Dim>>>(*derived, bbox, order); \
  }

  IMPL(biharmonic2d)
  IMPL(biharmonic3d)
  IMPL(cov_cauchy3);
  IMPL(cov_cauchy5);
  IMPL(cov_cauchy7);
  IMPL(cov_cauchy9);
  IMPL(cov_cubic);
  IMPL(cov_exponential);
  IMPL(cov_gaussian);
  IMPL(cov_spherical);
  IMPL(cov_spheroidal3);
  IMPL(cov_spheroidal5);
  IMPL(cov_spheroidal7);
  IMPL(cov_spheroidal9);
  IMPL(inverse_multiquadric1);
  IMPL(multiquadric1);
  IMPL(multiquadric3);
  IMPL(triharmonic2d)
  IMPL(triharmonic3d)

#undef IMPL

  throw std::invalid_argument("RBF type not supported.");
}

template FmmGenericEvaluatorPtr<1> make_fmm_evaluator<1>(const RbfPtr<1>& rbf,
                                                         const bboxNd<1>& bbox, int order);

template FmmGenericEvaluatorPtr<2> make_fmm_evaluator<2>(const RbfPtr<2>& rbf,
                                                         const bboxNd<2>& bbox, int order);

template FmmGenericEvaluatorPtr<3> make_fmm_evaluator<3>(const RbfPtr<3>& rbf,
                                                         const bboxNd<3>& bbox, int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_evaluator(const RbfPtr<Dim>& rbf,
                                                        const bboxNd<Dim>& bbox, int order) {
  auto* base = rbf.get();

#define IMPL(RBF_NAME)                                                                     \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                \
    return std::make_unique<fmm_gradient_evaluator<RBF_NAME<Dim>>>(*derived, bbox, order); \
  }

  IMPL(biharmonic2d)
  IMPL(biharmonic3d)
  IMPL(cov_cauchy3);
  IMPL(cov_cauchy5);
  IMPL(cov_cauchy7);
  IMPL(cov_cauchy9);
  IMPL(cov_cubic);
  IMPL(cov_exponential);
  IMPL(cov_gaussian);
  IMPL(cov_spherical);
  IMPL(cov_spheroidal3);
  IMPL(cov_spheroidal5);
  IMPL(cov_spheroidal7);
  IMPL(cov_spheroidal9);
  IMPL(inverse_multiquadric1);
  IMPL(multiquadric1);
  IMPL(multiquadric3);
  IMPL(triharmonic2d)
  IMPL(triharmonic3d)

#undef IMPL

  throw std::invalid_argument("RBF type not supported.");
}

template FmmGenericEvaluatorPtr<1> make_fmm_gradient_evaluator<1>(const RbfPtr<1>& rbf,
                                                                  const bboxNd<1>& bbox, int order);

template FmmGenericEvaluatorPtr<2> make_fmm_gradient_evaluator<2>(const RbfPtr<2>& rbf,
                                                                  const bboxNd<2>& bbox, int order);

template FmmGenericEvaluatorPtr<3> make_fmm_gradient_evaluator<3>(const RbfPtr<3>& rbf,
                                                                  const bboxNd<3>& bbox, int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_gradient_transpose_evaluator(const RbfPtr<Dim>& rbf,
                                                                  const bboxNd<Dim>& bbox,
                                                                  int order) {
  auto* base = rbf.get();

#define IMPL(RBF_NAME)                                                                       \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                  \
    return std::make_unique<fmm_gradient_transpose_evaluator<RBF_NAME<Dim>>>(*derived, bbox, \
                                                                             order);         \
  }

  IMPL(biharmonic2d)
  IMPL(biharmonic3d)
  IMPL(cov_cauchy3);
  IMPL(cov_cauchy5);
  IMPL(cov_cauchy7);
  IMPL(cov_cauchy9);
  IMPL(cov_cubic);
  IMPL(cov_exponential);
  IMPL(cov_gaussian);
  IMPL(cov_spherical);
  IMPL(cov_spheroidal3);
  IMPL(cov_spheroidal5);
  IMPL(cov_spheroidal7);
  IMPL(cov_spheroidal9);
  IMPL(inverse_multiquadric1);
  IMPL(multiquadric1);
  IMPL(multiquadric3);
  IMPL(triharmonic2d)
  IMPL(triharmonic3d)

#undef IMPL

  throw std::invalid_argument("RBF type not supported.");
}

template FmmGenericEvaluatorPtr<1> make_fmm_gradient_transpose_evaluator<1>(const RbfPtr<1>& rbf,
                                                                            const bboxNd<1>& bbox,
                                                                            int order);

template FmmGenericEvaluatorPtr<2> make_fmm_gradient_transpose_evaluator<2>(const RbfPtr<2>& rbf,
                                                                            const bboxNd<2>& bbox,
                                                                            int order);

template FmmGenericEvaluatorPtr<3> make_fmm_gradient_transpose_evaluator<3>(const RbfPtr<3>& rbf,
                                                                            const bboxNd<3>& bbox,
                                                                            int order);

template <int Dim>
FmmGenericEvaluatorPtr<Dim> make_fmm_hessian_evaluator(const RbfPtr<Dim>& rbf,
                                                       const bboxNd<Dim>& bbox, int order) {
  auto* base = rbf.get();

#define IMPL(RBF_NAME)                                                                    \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                               \
    return std::make_unique<fmm_hessian_evaluator<RBF_NAME<Dim>>>(*derived, bbox, order); \
  }

  IMPL(biharmonic2d)
  IMPL(biharmonic3d)
  IMPL(cov_cauchy3);
  IMPL(cov_cauchy5);
  IMPL(cov_cauchy7);
  IMPL(cov_cauchy9);
  IMPL(cov_cubic);
  IMPL(cov_exponential);
  IMPL(cov_gaussian);
  IMPL(cov_spherical);
  IMPL(cov_spheroidal3);
  IMPL(cov_spheroidal5);
  IMPL(cov_spheroidal7);
  IMPL(cov_spheroidal9);
  IMPL(inverse_multiquadric1);
  IMPL(multiquadric1);
  IMPL(multiquadric3);
  IMPL(triharmonic2d)
  IMPL(triharmonic3d)

#undef IMPL

  throw std::invalid_argument("RBF type not supported.");
}

template FmmGenericEvaluatorPtr<1> make_fmm_hessian_evaluator<1>(const RbfPtr<1>& rbf,
                                                                 const bboxNd<1>& bbox, int order);

template FmmGenericEvaluatorPtr<2> make_fmm_hessian_evaluator<2>(const RbfPtr<2>& rbf,
                                                                 const bboxNd<2>& bbox, int order);

template FmmGenericEvaluatorPtr<3> make_fmm_hessian_evaluator<3>(const RbfPtr<3>& rbf,
                                                                 const bboxNd<3>& bbox, int order);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_symmetric_evaluator(const RbfPtr<Dim>& rbf,
                                                                  const bboxNd<Dim>& bbox,
                                                                  int order) {
  auto* base = rbf.get();

#define IMPL(RBF_NAME)                                                                      \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                 \
    return std::make_unique<fmm_symmetric_evaluator<RBF_NAME<Dim>>>(*derived, bbox, order); \
  }

  IMPL(biharmonic2d)
  IMPL(biharmonic3d)
  IMPL(cov_cauchy3);
  IMPL(cov_cauchy5);
  IMPL(cov_cauchy7);
  IMPL(cov_cauchy9);
  IMPL(cov_cubic);
  IMPL(cov_exponential);
  IMPL(cov_gaussian);
  IMPL(cov_spherical);
  IMPL(cov_spheroidal3);
  IMPL(cov_spheroidal5);
  IMPL(cov_spheroidal7);
  IMPL(cov_spheroidal9);
  IMPL(inverse_multiquadric1);
  IMPL(multiquadric1);
  IMPL(multiquadric3);
  IMPL(triharmonic2d)
  IMPL(triharmonic3d)

#undef IMPL

  throw std::runtime_error("RBF type not supported.");
}

template FmmGenericSymmetricEvaluatorPtr<1> make_fmm_symmetric_evaluator<1>(const RbfPtr<1>& rbf,
                                                                            const bboxNd<1>& bbox,
                                                                            int order);

template FmmGenericSymmetricEvaluatorPtr<2> make_fmm_symmetric_evaluator<2>(const RbfPtr<2>& rbf,
                                                                            const bboxNd<2>& bbox,
                                                                            int order);

template FmmGenericSymmetricEvaluatorPtr<3> make_fmm_symmetric_evaluator<3>(const RbfPtr<3>& rbf,
                                                                            const bboxNd<3>& bbox,
                                                                            int order);

template <int Dim>
FmmGenericSymmetricEvaluatorPtr<Dim> make_fmm_hessian_symmetric_evaluator(const RbfPtr<Dim>& rbf,
                                                                          const bboxNd<Dim>& bbox,
                                                                          int order) {
  auto* base = rbf.get();

#define IMPL(RBF_NAME)                                                                      \
  if (auto* derived = dynamic_cast<RBF_NAME<Dim>*>(base)) {                                 \
    return std::make_unique<fmm_hessian_symmetric_evaluator<RBF_NAME<Dim>>>(*derived, bbox, \
                                                                            order);         \
  }

  IMPL(biharmonic2d)
  IMPL(biharmonic3d)
  IMPL(cov_cauchy3);
  IMPL(cov_cauchy5);
  IMPL(cov_cauchy7);
  IMPL(cov_cauchy9);
  IMPL(cov_cubic);
  IMPL(cov_exponential);
  IMPL(cov_gaussian);
  IMPL(cov_spherical);
  IMPL(cov_spheroidal3);
  IMPL(cov_spheroidal5);
  IMPL(cov_spheroidal7);
  IMPL(cov_spheroidal9);
  IMPL(inverse_multiquadric1);
  IMPL(multiquadric1);
  IMPL(multiquadric3);
  IMPL(triharmonic2d)
  IMPL(triharmonic3d)

#undef IMPL

  throw std::runtime_error("RBF type not supported.");
}

template FmmGenericSymmetricEvaluatorPtr<1> make_fmm_hessian_symmetric_evaluator<1>(
    const RbfPtr<1>& rbf, const bboxNd<1>& bbox, int order);

template FmmGenericSymmetricEvaluatorPtr<2> make_fmm_hessian_symmetric_evaluator<2>(
    const RbfPtr<2>& rbf, const bboxNd<2>& bbox, int order);

template FmmGenericSymmetricEvaluatorPtr<3> make_fmm_hessian_symmetric_evaluator<3>(
    const RbfPtr<3>& rbf, const bboxNd<3>& bbox, int order);

}  // namespace polatory::fmm
