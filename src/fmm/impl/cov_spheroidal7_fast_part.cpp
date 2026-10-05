#include <jizai/rbf/cov_spheroidal7.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovSpheroidal7FastPart);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovSpheroidal7FastPart);

}  // namespace jizai::fmm
