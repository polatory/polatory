// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <stdexcept>
#include <string>

namespace polatory {
namespace common {

class unsupported_method : public std::runtime_error {
public:
   explicit unsupported_method(const std::string& n)
      : std::runtime_error(std::string("unsupported method (") + n + ")")
   {
   }
};

} // namespace common
} // namespace polatory
