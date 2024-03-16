#include <polatory/rbf/cov_cubic.hpp>

#include "../direct_evaluator.hpp"
#include "../direct_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::cov_cubic);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::cov_cubic);

}  // namespace polatory::fmm
