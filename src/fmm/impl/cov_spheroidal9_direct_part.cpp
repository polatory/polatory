#include <jizai/rbf/cov_spheroidal9.hpp>

#include "../direct_evaluator.hpp"
#include "../direct_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal9DirectPart);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal9DirectPart);

}  // namespace jizai::fmm
