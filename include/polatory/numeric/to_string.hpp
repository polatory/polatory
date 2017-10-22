// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>

namespace polatory {
namespace numeric {

namespace detail {

template <class Floating>
struct lexical_cast;

template <>
struct lexical_cast<float> {
  float operator()(const std::string& str) const {
    return std::stof(str);
  }
};

template <>
struct lexical_cast<double> {
  double operator()(const std::string& str) const {
    return std::stod(str);
  }
};

template <class Floating>
struct format;

template <>
struct format<float> {
  static const char *shorthand() {
    return "%.7g";
  }

  static const char *complete() {
    return "%.9g";
  }
};

template <>
struct format<double> {
  static const char *shorthand() {
    return "%.15g";
  }

  static const char *complete() {
    return "%.17g";
  }
};

} // namespace detail

// Performance comparison between std::stringstream and std::sprintf:
// http://www.boost.org/doc/libs/1_65_1/doc/html/boost_lexical_cast/performance.html
template <class Floating, typename std::enable_if<std::is_floating_point<Floating>::value, std::nullptr_t>::type = nullptr>
std::string to_string(Floating arg) {
  char str[32];

  if (std::isnan(arg))
    return "nan";

  if (std::isinf(arg))
    return std::signbit(arg) ? "-inf" : "inf";

  std::sprintf(str, detail::format<Floating>::shorthand(), arg);
  Floating arg_rep = detail::lexical_cast<Floating>()(str);
  if (arg == arg_rep)
    return str;

  std::sprintf(str, detail::format<Floating>::complete(), arg);
  return str;
}

} // namespace numeric
} // namespace polatory
