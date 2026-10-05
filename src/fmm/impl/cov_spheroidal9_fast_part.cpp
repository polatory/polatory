#include <jizai/rbf/cov_spheroidal9.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal9FastPart);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal9FastPart);

}  // namespace jizai::fmm
