#include <jizai/rbf/polyharmonic_even.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::Triharmonic2D);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::Triharmonic2D);

}  // namespace jizai::fmm
