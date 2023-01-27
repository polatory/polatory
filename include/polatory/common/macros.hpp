#pragma once

#include <cassert>

#define POLATORY_ASSERT(X) \
  assert(                  \
      X)  // NOLINT(cppcoreguidelines-macro-usage,cppcoreguidelines-pro-bounds-array-to-pointer-decay)

#define POLATORY_NEVER_REACH() \
  assert(                      \
      false)  // NOLINT(cert-dcl03-c,cppcoreguidelines-macro-usage,cppcoreguidelines-pro-bounds-array-to-pointer-decay,misc-static-assert)

#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#define POLATORY_LIKELY(X) __builtin_expect(!!(X), 1)    // NOLINT(cppcoreguidelines-macro-usage)
#define POLATORY_UNLIKELY(X) __builtin_expect(!!(X), 0)  // NOLINT(cppcoreguidelines-macro-usage)
#else
#define POLATORY_LIKELY(X) (X)
#define POLATORY_UNLIKELY(X) (X)
#endif
