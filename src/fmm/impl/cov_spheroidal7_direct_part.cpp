#include <jizai/rbf/cov_spheroidal7.hpp>

#include "../direct_evaluator.hpp"
#include "../direct_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal7DirectPart);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal7DirectPart);

}  // namespace jizai::fmm
