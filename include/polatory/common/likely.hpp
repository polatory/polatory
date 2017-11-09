// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#if defined(__GNUC__) || defined(__INTEL_COMPILER)
#  define LIKELY(X) __builtin_expect(!!(X), 1)
#  define UNLIKELY(X) __builtin_expect(!!(X), 0)
#else
#  define LIKELY(X) (X)
#  define UNLIKELY(X) (X)
#endif
