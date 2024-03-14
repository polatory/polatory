#include <polatory/rbf/cov_cubic.hpp>

#include "../direct_evaluator.hpp"
#include "../direct_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::cov_cubic);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::cov_cubic);

}  // namespace polatory::fmm
