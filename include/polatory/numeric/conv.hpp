#pragma once

#include <fast_float/fast_float.h>

#include <array>
#include <charconv>
#include <limits>
#include <string>

namespace polatory::numeric {

inline double to_double(const std::string& str) {
  double result{std::numeric_limits<double>::quiet_NaN()};
  fast_float::from_chars(str.data(), str.data() + str.size(), result);
  return result;
}

inline float to_float(const std::string& str) {
  float result{std::numeric_limits<float>::quiet_NaN()};
  fast_float::from_chars(str.data(), str.data() + str.size(), result);
  return result;
}

template <class T>
std::string to_string(T value) {
  std::array<char, 32> buffer{};
  auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  return std::string(buffer.data(), ptr);
}

}  // namespace polatory::numeric
