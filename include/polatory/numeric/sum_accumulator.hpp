// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cstddef>
#include <type_traits>

namespace polatory {
namespace numeric {

template <class Floating, typename std::enable_if<std::is_floating_point<Floating>::value, std::nullptr_t>::type = nullptr>
class kahan_sum_accumulator {
public:
  kahan_sum_accumulator()
    : sum_()
    , correction_() {
  }

  Floating get() const {
    return sum_;
  }

  kahan_sum_accumulator& operator+=(Floating f) {
    auto summand = f + correction_;
    auto next_sum = sum_ + summand;
    correction_ = summand - (next_sum - sum_);
    sum_ = next_sum;
    return *this;
  }

private:
  Floating sum_;
  Floating correction_;
};

template <class Floating, typename std::enable_if<std::is_floating_point<Floating>::value, std::nullptr_t>::type = nullptr>
class knuth_sum_accumulator {
public:
  knuth_sum_accumulator()
    : sum_()
    , correction_() {
  }

  Floating get() const {
    return sum_;
  }

  knuth_sum_accumulator& operator+=(Floating f) {
    auto summand = f + correction_;
    auto next_sum = sum_ + summand;
    auto up = next_sum - summand;
    auto vpp = next_sum - up;
    correction_ = (sum_ - up) + (summand - vpp);
    sum_ = next_sum;
    return *this;
  }

private:
  Floating sum_;
  Floating correction_;
};

}  // namespace numeric
}  // namespace polatory
