#pragma once

#include <cassert>

#define POLATORY_ASSERT(X) \
  assert(                  \
      X)  // NOLINT(cppcoreguidelines-macro-usage,cppcoreguidelines-pro-bounds-array-to-pointer-decay)

#define POLATORY_NEVER_REACH() \
  assert(                      \
      false)  // NOLINT(cert-dcl03-c,cppcoreguidelines-macro-usage,cppcoreguidelines-pro-bounds-array-to-pointer-decay,misc-static-assert)
