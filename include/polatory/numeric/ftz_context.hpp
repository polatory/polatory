// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <pmmintrin.h>
#include <xmmintrin.h>
#include <cmath>
#include <limits>

#include <boost/core/noncopyable.hpp>

namespace polatory {
namespace numeric {

class ftz_context : private boost::noncopyable {
public:
  ftz_context()
    : ftz_mode_backup_(_MM_GET_FLUSH_ZERO_MODE())
    , daz_mode_backup_(_MM_GET_DENORMALS_ZERO_MODE()) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }

  ~ftz_context() {
    _MM_SET_FLUSH_ZERO_MODE(ftz_mode_backup_);
    _MM_SET_DENORMALS_ZERO_MODE(daz_mode_backup_);
  }

  static bool daz_is_active() {
    volatile double double_denorm_min = std::numeric_limits<double>::denorm_min();
    return std::pow(2, 52) * double_denorm_min == 0.0;
  }

  static bool ftz_is_active() {
    volatile double double_min = std::numeric_limits<double>::min();
    return std::pow(2, -52) * double_min == 0.0;
  }

private:
  decltype(_MM_FLUSH_ZERO_ON) ftz_mode_backup_;
  decltype(_MM_DENORMALS_ZERO_ON) daz_mode_backup_;
};

} // namespace numeric
} // namespace polatory
