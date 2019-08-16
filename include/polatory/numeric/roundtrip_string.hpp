// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>

#include <boost/lexical_cast.hpp>

namespace polatory {
namespace numeric {

namespace detail {

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

}  // namespace detail

inline
double to_double(const std::string& str) {
  return boost::lexical_cast<double>(str);
}

inline
float to_float(const std::string& str) {
  return boost::lexical_cast<float>(str);
}

// Performance comparison between std::stringstream and std::sprintf:
// http://www.boost.org/doc/libs/1_65_1/doc/html/boost_lexical_cast/performance.html
template <class Floating, typename std::enable_if<std::is_floating_point<Floating>::value, std::nullptr_t>::type = nullptr>
std::string to_string(Floating arg) {
  static constexpr size_t str_size = 32;
  char str[str_size];  // NOLINT(runtime/arrays)

  if (std::isnan(arg))
    return "nan";

  if (std::isinf(arg))
    return std::signbit(arg) ? "-inf" : "inf";

  std::snprintf(str, str_size, detail::format<Floating>::shorthand(), arg);  // NOLINT(cppcoreguidelines-pro-type-vararg)
  auto arg_rep = boost::lexical_cast<Floating>(str);
  if (arg == arg_rep)
    return str;

  std::snprintf(str, str_size, detail::format<Floating>::complete(), arg);  // NOLINT(cppcoreguidelines-pro-type-vararg)
  return str;
}

}  // namespace numeric
}  // namespace polatory
