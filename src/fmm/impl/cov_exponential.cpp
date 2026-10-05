#include <jizai/rbf/cov_exponential.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovExponential);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovExponential);

}  // namespace jizai::fmm
