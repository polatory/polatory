#pragma once

#include <format>
#include <polatory/rbf/cov_cubic.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_gaussian.hpp>
#include <polatory/rbf/cov_generalized_cauchy3.hpp>
#include <polatory/rbf/cov_generalized_cauchy5.hpp>
#include <polatory/rbf/cov_generalized_cauchy7.hpp>
#include <polatory/rbf/cov_generalized_cauchy9.hpp>
#include <polatory/rbf/cov_spherical.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/inverse_multiquadric.hpp>
#include <polatory/rbf/multiquadric.hpp>
#include <polatory/rbf/polyharmonic_even.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/rbf/rbf_proxy.hpp>
#include <stdexcept>

namespace polatory::rbf {

template <int Dim>
rbf_proxy<Dim> make_rbf(const std::string& name, const std::vector<double>& params) {
#define POLATORY_CASE(RBF_NAME)                 \
  if (name == RBF_NAME<Dim>::Rbf::kShortName) { \
    return RBF_NAME<Dim>(params);               \
  }

  POLATORY_CASE(biharmonic2d);
  POLATORY_CASE(biharmonic3d);
  POLATORY_CASE(cov_cubic);
  POLATORY_CASE(cov_exponential);
  POLATORY_CASE(cov_gaussian);
  POLATORY_CASE(cov_generalized_cauchy3);
  POLATORY_CASE(cov_generalized_cauchy5);
  POLATORY_CASE(cov_generalized_cauchy7);
  POLATORY_CASE(cov_generalized_cauchy9);
  POLATORY_CASE(inverse_multiquadric1);
  POLATORY_CASE(multiquadric1);
  POLATORY_CASE(multiquadric3);
  POLATORY_CASE(cov_spherical);
  POLATORY_CASE(cov_spheroidal3);
  POLATORY_CASE(cov_spheroidal5);
  POLATORY_CASE(cov_spheroidal7);
  POLATORY_CASE(cov_spheroidal9);
  POLATORY_CASE(triharmonic2d);
  POLATORY_CASE(triharmonic3d);

#undef POLATORY_CASE

  throw std::runtime_error(std::format("unknown RBF name: '{}'", name));
}

}  // namespace polatory::rbf
