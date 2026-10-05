#pragma once

#include <jizai/common/macros.hpp>
#include <jizai/types.hpp>

namespace jizai::polynomial {

template <int Dim>
class PolynomialBasisBase {
 public:
  static constexpr int kDim = Dim;

  explicit PolynomialBasisBase(int degree) : degree_(degree) { JIZAI_ASSERT(degree >= 0); }

  virtual ~PolynomialBasisBase() = default;

  PolynomialBasisBase(const PolynomialBasisBase&) = delete;
  PolynomialBasisBase(PolynomialBasisBase&&) = delete;
  PolynomialBasisBase& operator=(const PolynomialBasisBase&) = delete;
  PolynomialBasisBase& operator=(PolynomialBasisBase&&) = delete;

  Index basis_size() const { return basis_size(degree_); }

  int degree() const { return degree_; }

  static Index basis_size(int degree) {
    if (degree < 0) {
      return 0;
    }

    auto k = Index{degree} + 1;
    switch (kDim) {
      case 1:
        return k;
      case 2:
        return k * (k + 1) / 2;
      case 3:
        return k * (k + 1) * (k + 2) / 6;
      default:
        JIZAI_UNREACHABLE();
        break;
    }

    return 0;
  }

 private:
  const int degree_;
};

}  // namespace jizai::polynomial
