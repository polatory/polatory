#include <jizai/rbf/cov_spheroidal5.hpp>

#include "../direct_evaluator.hpp"
#include "../direct_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal5DirectPart);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal5DirectPart);

}  // namespace jizai::fmm
