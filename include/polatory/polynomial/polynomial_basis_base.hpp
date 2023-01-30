#pragma once

#include <polatory/common/macros.hpp>
#include <polatory/types.hpp>

namespace polatory::polynomial {

class polynomial_basis_base {
 public:
  explicit polynomial_basis_base(int dimension, int degree)
      : dimension_(dimension), degree_(degree) {
    POLATORY_ASSERT(dimension >= 1 && dimension <= 3);
    POLATORY_ASSERT(degree >= 0);
  }

  virtual ~polynomial_basis_base() = default;

  polynomial_basis_base(const polynomial_basis_base&) = delete;
  polynomial_basis_base(polynomial_basis_base&&) = delete;
  polynomial_basis_base& operator=(const polynomial_basis_base&) = delete;
  polynomial_basis_base& operator=(polynomial_basis_base&&) = delete;

  index_t basis_size() const { return basis_size(dimension_, degree_); }

  int degree() const { return degree_; }

  int dimension() const { return dimension_; }

  static index_t basis_size(int dimension, int degree) {
    if (degree < 0) {
      return 0;
    }
    POLATORY_ASSERT(dimension >= 1 && dimension <= 3);

    auto k = static_cast<index_t>(degree) + 1;
    switch (dimension) {
      case 1:
        return k;
      case 2:
        return k * (k + 1) / 2;
      case 3:
        return k * (k + 1) * (k + 2) / 6;
      default:
        POLATORY_UNREACHABLE();
        break;
    }

    return 0;
  }

 private:
  const int dimension_;
  const int degree_;
};

}  // namespace polatory::polynomial
