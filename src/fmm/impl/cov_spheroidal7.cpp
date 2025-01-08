#include <polatory/rbf/cov_spheroidal7.hpp>

#include "../spheroidal_evaluator.hpp"
#include "../spheroidal_symmetric_evaluator.hpp"

namespace polatory::fmm {

EXTERN_FMM_EVALUATORS(rbf::internal::CovSpheroidal7DirectPart)
EXTERN_FMM_EVALUATORS(rbf::internal::CovSpheroidal7FastPart)
IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal7);

EXTERN_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal7DirectPart)
EXTERN_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal7FastPart)
IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal7);

}  // namespace polatory::fmm
