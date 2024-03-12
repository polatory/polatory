#include <polatory/rbf/polyharmonic_even.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::triharmonic2d);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::triharmonic2d);

}  // namespace polatory::fmm
