#pragma once

#include <memory>
#include <polatory/rbf/rbf_base.hpp>

namespace polatory::rbf {

template <class Rbf>
RbfPtr<Rbf::kDim> make_rbf(const std::vector<double>& params) {
  return std::make_unique<Rbf>(params);
}

}  // namespace polatory::rbf
