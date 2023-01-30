#pragma once

#include <cassert>

namespace polatory {

inline void POLATORY_ASSERT([[maybe_unused]] bool x) { assert(x); }

inline void POLATORY_UNREACHABLE() { assert(false); }

}  // namespace polatory
