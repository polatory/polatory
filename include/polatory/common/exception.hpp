// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <stdexcept>
#include <string>

namespace polatory {
namespace common {

class invalid_parameter : public std::runtime_error {
public:
  explicit invalid_parameter(const std::string& expected)
    : std::runtime_error(std::string("invalid parameter (expected: ") + expected + ")") {
  }
};

class unsupported_method : public std::runtime_error {
public:
  explicit unsupported_method(const std::string& n)
    : std::runtime_error(std::string("unsupported method (") + n + ")") {
  }
};

} // namespace common
} // namespace polatory
