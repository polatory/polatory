#include <jizai/rbf/cov_spheroidal9.hpp>

#include "../spheroidal_evaluator.hpp"
#include "../spheroidal_symmetric_evaluator.hpp"

namespace jizai::fmm {

EXTERN_FMM_EVALUATORS(rbf::internal::CovSpheroidal9DirectPart)
EXTERN_FMM_EVALUATORS(rbf::internal::CovSpheroidal9FastPart)
IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal9);

EXTERN_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal9DirectPart)
EXTERN_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal9FastPart)
IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal9);

}  // namespace jizai::fmm
