// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <stdexcept>
#include <string>

namespace polatory {
namespace common {

class invalid_argument : public std::logic_error {
public:
  explicit invalid_argument(const std::string& expected)
    : std::logic_error("Invalid argument (expected: " + expected + ").") {}
};

class not_supported : public std::logic_error {
public:
  explicit not_supported(const std::string& name)
    : std::logic_error(name + "is not supported.") {}
};

} // namespace common
} // namespace polatory
