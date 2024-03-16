#include <polatory/rbf/cov_cauchy9.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::cov_cauchy9);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::cov_cauchy9);

}  // namespace polatory::fmm
