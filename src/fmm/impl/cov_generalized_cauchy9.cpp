#include <polatory/rbf/cov_generalized_cauchy9.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovGeneralizedCauchy9);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovGeneralizedCauchy9);

}  // namespace polatory::fmm
