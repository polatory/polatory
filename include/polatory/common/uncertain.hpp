// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <polatory/common/macros.hpp>

namespace polatory {
namespace common {

template <class T>
class uncertain {
public:
  uncertain() = default;

  uncertain(T value)  // NOLINT(runtime/explicit)
    : certain_(true)
    , value_(value) {
  }

  T get() const {
    POLATORY_ASSERT(is_certain());
    return value_;
  }

  bool is_certain() const {
    return certain_;
  }

private:
  bool certain_{false};
  T value_{};
};

inline
uncertain<bool> operator!(uncertain<bool> a) {
  if (a.is_certain()) return !a.get();
  return {};
}

inline
uncertain<bool> operator&&(uncertain<bool> a, uncertain<bool> b) {
  if (a.is_certain()) {
    if (b.is_certain()) {
      return a.get() && b.get();
    }
    if (!a.get()) {
      return false;
    }
  } else if (b.is_certain()) {
    if (!b.get()) {
      return false;
    }
  }
  return {};
}

inline
uncertain<bool> operator||(uncertain<bool> a, uncertain<bool> b) {
  if (a.is_certain()) {
    if (b.is_certain()) {
      return a.get() || b.get();
    }
    if (a.get()) {
      return true;
    }
  } else if (b.is_certain()) {
    if (b.get()) {
      return true;
    }
  }
  return {};
}

inline
bool certainly(uncertain<bool> a) {
  return a.is_certain() && a.get();
}

inline
bool certainly_not(uncertain<bool> a) {
  return a.is_certain() && !a.get();
}

inline
bool possibly(uncertain<bool> a) {
  return !a.is_certain() || a.get();
}

inline
bool possibly_not(uncertain<bool> a) {
  return !a.is_certain() || !a.get();
}

}  // namespace common
}  // namespace polatory
