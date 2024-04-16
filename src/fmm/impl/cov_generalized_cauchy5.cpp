#include <polatory/rbf/cov_generalized_cauchy5.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::cov_generalized_cauchy5);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::cov_generalized_cauchy5);

}  // namespace polatory::fmm
