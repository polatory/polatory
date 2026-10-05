#include <jizai/rbf/cov_cubic.hpp>

#include "../direct_evaluator.hpp"
#include "../direct_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovCubic);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovCubic);

}  // namespace jizai::fmm
