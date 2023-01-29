#pragma once

#include <double-conversion/double-conversion.h>

#include <array>
#include <limits>
#include <string>
#include <type_traits>

namespace polatory::numeric {

namespace detail {

template <class Floating>
void to_string(Floating value, double_conversion::StringBuilder* builder);

template <>
inline void to_string(double value, double_conversion::StringBuilder* builder) {
  double_conversion::DoubleToStringConverter::EcmaScriptConverter().ToShortest(value, builder);
}

template <>
inline void to_string(float value, double_conversion::StringBuilder* builder) {
  double_conversion::DoubleToStringConverter::EcmaScriptConverter().ToShortestSingle(value,
                                                                                     builder);
}

}  // namespace detail

inline double to_double(const std::string& str) {
  double_conversion::StringToDoubleConverter converter(
      double_conversion::StringToDoubleConverter::ALLOW_TRAILING_JUNK |
          double_conversion::StringToDoubleConverter::ALLOW_CASE_INSENSITIVITY,
      std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), "inf",
      "nan");

  int processed_chars{};
  return converter.StringToDouble(str.data(), static_cast<int>(str.size()), &processed_chars);
}

inline float to_float(const std::string& str) {
  double_conversion::StringToDoubleConverter converter(
      double_conversion::StringToDoubleConverter::ALLOW_TRAILING_JUNK |
          double_conversion::StringToDoubleConverter::ALLOW_CASE_INSENSITIVITY,
      std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), "inf",
      "nan");

  int processed_chars{};
  return converter.StringToFloat(str.data(), static_cast<int>(str.size()), &processed_chars);
}

template <class Floating,
          std::enable_if_t<std::is_floating_point<Floating>::value, std::nullptr_t> = nullptr>
std::string to_string(Floating value) {
  static constexpr size_t buffer_size = 32;

  std::array<char, buffer_size> buffer{};
  double_conversion::StringBuilder builder(buffer.data(), buffer_size);

  detail::to_string(value, &builder);

  return builder.Finalize();
}

}  // namespace polatory::numeric
