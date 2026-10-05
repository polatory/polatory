#include <jizai/rbf/cov_generalized_cauchy5.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovGeneralizedCauchy5);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovGeneralizedCauchy5);

}  // namespace jizai::fmm
