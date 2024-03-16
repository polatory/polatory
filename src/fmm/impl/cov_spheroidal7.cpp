#include <polatory/rbf/cov_spheroidal7.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::cov_spheroidal7);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::cov_spheroidal7);

}  // namespace polatory::fmm
