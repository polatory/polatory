// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#  define POLATORY_LIKELY(X) __builtin_expect(!!(X), 1)  // NOLINT(cppcoreguidelines-macro-usage)
#  define POLATORY_UNLIKELY(X) __builtin_expect(!!(X), 0)  // NOLINT(cppcoreguidelines-macro-usage)
#else
#  define POLATORY_LIKELY(X) (X)
#  define POLATORY_UNLIKELY(X) (X)
#endif
