#include <jizai/rbf/cov_gaussian.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::CovGaussian);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::CovGaussian);

}  // namespace jizai::fmm
