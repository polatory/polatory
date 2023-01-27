#pragma once

#include <double-conversion/double-conversion.h>

#include <boost/lexical_cast.hpp>
#include <string>
#include <type_traits>

namespace polatory {
namespace numeric {

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

inline double to_double(const std::string& str) { return boost::lexical_cast<double>(str); }

inline float to_float(const std::string& str) { return boost::lexical_cast<float>(str); }

template <class Floating,
          std::enable_if_t<std::is_floating_point<Floating>::value, std::nullptr_t> = nullptr>
std::string to_string(Floating value) {
  static constexpr size_t buffer_size = 32;

  std::array<char, buffer_size> buffer{};
  double_conversion::StringBuilder builder(buffer.data(), buffer_size);

  detail::to_string(value, &builder);

  return builder.Finalize();
}

}  // namespace numeric
}  // namespace polatory
