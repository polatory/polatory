// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#if defined(_MSC_VER)
#  define LIKELY(X) (X)
#  define UNLIKELY(X) (X)
#elif defined(__GNUC__) || defined(__INTEL_COMPILER)
// Intel C++ Compiler (for Linux and OS X) and Clang also define __GNUC__
#  define LIKELY(X) __builtin_expect(!!(X), 1)
#  define UNLIKELY(X) __builtin_expect(!!(X), 0)
#else
#  define LIKELY(X) (X)
#  define UNLIKELY(X) (X)
#endif
