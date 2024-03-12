#include <polatory/rbf/multiquadric.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::multiquadric1);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::multiquadric1);

}  // namespace polatory::fmm
