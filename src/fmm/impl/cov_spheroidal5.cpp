#include <polatory/rbf/cov_spheroidal5.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::cov_spheroidal5);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::cov_spheroidal5);

}  // namespace polatory::fmm
