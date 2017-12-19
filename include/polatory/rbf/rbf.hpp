// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <memory>

#include <polatory/rbf/rbf_kernel.hpp>

namespace polatory {
namespace rbf {

class rbf {
public:
  rbf() {
  }

  rbf(const rbf_kernel& kernel) // NOLINT(runtime/explicit)
    : kern_(kernel.clone()) {
  }

  const rbf_kernel& get() const {
    return *kern_;
  }

  rbf_kernel& get() {
    return *kern_;
  }

private:
  std::shared_ptr<rbf_kernel> kern_;
};

} // namespace rbf
} // namespace polatory
