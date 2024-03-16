#include <polatory/rbf/polyharmonic_odd.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::triharmonic3d);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::triharmonic3d);

}  // namespace polatory::fmm
