#include <polatory/rbf/cov_spheroidal5.hpp>

#include "../spheroidal_evaluator.hpp"
#include "../spheroidal_symmetric_evaluator.hpp"

namespace polatory::fmm {

EXTERN_FMM_EVALUATORS(rbf::internal::CovSpheroidal5DirectPart)
EXTERN_FMM_EVALUATORS(rbf::internal::CovSpheroidal5FastPart)
IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal5);

EXTERN_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal5DirectPart)
EXTERN_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal5FastPart)
IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal5);

}  // namespace polatory::fmm
