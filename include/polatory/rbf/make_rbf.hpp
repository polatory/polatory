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
Rbf<Dim> make_rbf(const std::string& name, const std::vector<double>& params) {
#define POLATORY_CASE(RBF_NAME)            \
  if (name == RBF_NAME<Dim>::kShortName) { \
    return RBF_NAME<Dim>(params);          \
  }

  POLATORY_CASE(Biharmonic2D);
  POLATORY_CASE(Biharmonic3D);
  POLATORY_CASE(CovCubic);
  POLATORY_CASE(CovExponential);
  POLATORY_CASE(CovGaussian);
  POLATORY_CASE(CovGeneralizedCauchy3);
  POLATORY_CASE(CovGeneralizedCauchy5);
  POLATORY_CASE(CovGeneralizedCauchy7);
  POLATORY_CASE(CovGeneralizedCauchy9);
  POLATORY_CASE(InverseMultiquadric1);
  POLATORY_CASE(Multiquadric1);
  POLATORY_CASE(Multiquadric3);
  POLATORY_CASE(CovSpherical);
  POLATORY_CASE(CovSpheroidal3);
  POLATORY_CASE(CovSpheroidal5);
  POLATORY_CASE(CovSpheroidal7);
  POLATORY_CASE(CovSpheroidal9);
  POLATORY_CASE(Triharmonic2D);
  POLATORY_CASE(Triharmonic3D);

#undef POLATORY_CASE

  throw std::runtime_error(std::format("unknown RBF name: '{}'", name));
}

}  // namespace polatory::rbf
