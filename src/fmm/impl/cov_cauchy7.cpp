#include <polatory/rbf/cov_cauchy7.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::cov_cauchy7);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::cov_cauchy7);

}  // namespace polatory::fmm
