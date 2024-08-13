#include <polatory/rbf/cov_spherical.hpp>

#include "../direct_evaluator.hpp"
#include "../direct_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpherical);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpherical);

}  // namespace polatory::fmm
