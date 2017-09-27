// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

namespace polatory {
namespace polynomial {

class basis_base {
  const int deg;
  const size_t dim;

public:
  explicit basis_base(int degree)
    : deg(degree)
    , dim(dimension(degree)) {
    assert(degree >= 0);
  }

  virtual ~basis_base() {}

  // Degree of a polynomial.
  int degree() const {
    return deg;
  }

  // Size of the basis (degree of freedom).
  size_t dimension() const {
    return dimension(deg);
  }

  static size_t dimension(int degree) {
    if (degree < 0) return 0;

    size_t k = degree + 1;
    return k * (k + 1) * (k + 2) / 6;
  }
};

} // namespace polynomial
} // namespace polatory
