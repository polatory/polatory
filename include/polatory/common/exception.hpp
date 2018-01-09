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

class io_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class not_supported : public std::logic_error {
public:
  explicit not_supported(const std::string& what)
    : std::logic_error(what + " is not supported.") {}
};

}  // namespace common
}  // namespace polatory
