#include <polatory/rbf/cov_spheroidal3.hpp>

#include "../spheroidal_evaluator.hpp"
#include "../spheroidal_symmetric_evaluator.hpp"

namespace polatory::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal3);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal3);

}  // namespace polatory::fmm
