#include <polatory/rbf/cov_generalized_cauchy7.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovGeneralizedCauchy7);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovGeneralizedCauchy7);

}  // namespace polatory::fmm
