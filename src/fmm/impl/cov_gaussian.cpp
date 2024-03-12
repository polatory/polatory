#include <polatory/rbf/cov_gaussian.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::cov_gaussian);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::cov_gaussian);

}  // namespace polatory::fmm
