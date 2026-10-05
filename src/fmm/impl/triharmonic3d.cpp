#include <jizai/rbf/polyharmonic_odd.hpp>

#include "../fmm_evaluator.hpp"
#include "../fmm_symmetric_evaluator.hpp"

namespace jizai::fmm {

IMPLEMENT_FMM_EVALUATORS(rbf::internal::Triharmonic3D);

IMPLEMENT_FMM_SYMMETRIC_EVALUATORS(rbf::internal::Triharmonic3D);

}  // namespace jizai::fmm
