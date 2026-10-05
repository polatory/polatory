#include <jizai/rbf/cov_spheroidal3.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal3FastPart);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal3FastPart);

}  // namespace jizai::fmm
