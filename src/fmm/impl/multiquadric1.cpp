#include <polatory/rbf/multiquadric.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::Multiquadric1);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::Multiquadric1);

}  // namespace polatory::fmm
