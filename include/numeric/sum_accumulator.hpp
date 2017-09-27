// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>

namespace polatory {
namespace numeric {

template <typename T>
class kahan_sum_accumulator {
  T sum;
  T correction;

public:
  kahan_sum_accumulator()
    : sum()
    , correction() {
  }

  T get() const {
    return sum;
  }

  kahan_sum_accumulator& operator+=(T d) {
    auto summand = d + correction;
    auto next_sum = sum + summand;
    correction = summand - (next_sum - sum);
    sum = std::move(next_sum);
    return *this;
  }
};

template <typename T>
class knuth_sum_accumulator {
  T sum;
  T correction;

public:
  knuth_sum_accumulator()
    : sum()
    , correction() {
  }

  T get() const {
    return sum;
  }

  knuth_sum_accumulator& operator+=(T d) {
    auto u = std::move(sum);
    auto v = d + correction;
    auto uv = u + v;
    auto up = uv - v;
    auto vpp = uv - up;
    correction = (u - up) + (v - vpp);
    sum = std::move(uv);
    return *this;
  }
};

} // namespace numeric
} // namespace polatory
