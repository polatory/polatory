#include <polatory/rbf/cov_exponential.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovExponential);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovExponential);

}  // namespace polatory::fmm
