// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cassert>

namespace polatory {
namespace polynomial {

class basis_base {
public:
  explicit basis_base(int dimension, int degree)
    : dimension_(dimension)
    , degree_(degree)
    , basis_size_(basis_size(dimension, degree)) {
    assert(dimension >= 1 && dimension <= 3);
    assert(degree >= 0);
  }

  virtual ~basis_base() {}

  size_t basis_size() const {
    return basis_size(dimension_, degree_);
  }

  int degree() const {
    return degree_;
  }

  int dimension() const {
    return dimension_;
  }

  static size_t basis_size(int dimension, int degree) {
    if (degree < 0) return 0;
    assert(dimension >= 1 && dimension <= 3);

    size_t k = degree + 1;
    switch (dimension) {
    case 1:
      return k;
    case 2:
      return k * (k + 1) / 2;
    case 3:
      return k * (k + 1) * (k + 2) / 6;
    default:
      assert(false);
      break;
    }

    return 0;
  }

private:
  const int dimension_;
  const int degree_;
  const size_t basis_size_;
};

} // namespace polynomial
} // namespace polatory
