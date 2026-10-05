#pragma once

#include <format>
#include <jizai/rbf/cov_cubic.hpp>
#include <jizai/rbf/cov_exponential.hpp>
#include <jizai/rbf/cov_gaussian.hpp>
#include <jizai/rbf/cov_generalized_cauchy3.hpp>
#include <jizai/rbf/cov_generalized_cauchy5.hpp>
#include <jizai/rbf/cov_generalized_cauchy7.hpp>
#include <jizai/rbf/cov_generalized_cauchy9.hpp>
#include <jizai/rbf/cov_spherical.hpp>
#include <jizai/rbf/cov_spheroidal3.hpp>
#include <jizai/rbf/cov_spheroidal5.hpp>
#include <jizai/rbf/cov_spheroidal7.hpp>
#include <jizai/rbf/cov_spheroidal9.hpp>
#include <jizai/rbf/polyharmonic_even.hpp>
#include <jizai/rbf/polyharmonic_odd.hpp>
#include <jizai/rbf/rbf.hpp>
#include <stdexcept>

namespace jizai::rbf {

template <int Dim>
Rbf<Dim> make_rbf(const std::string& name, const std::vector<double>& params) {
#define JIZAI_CASE(RBF_NAME)               \
  if (name == RBF_NAME<Dim>::kShortName) { \
    return RBF_NAME<Dim>(params);          \
  }

  JIZAI_CASE(Biharmonic2D);
  JIZAI_CASE(Biharmonic3D);
  JIZAI_CASE(CovCubic);
  JIZAI_CASE(CovExponential);
  JIZAI_CASE(CovGaussian);
  JIZAI_CASE(CovGeneralizedCauchy3);
  JIZAI_CASE(CovGeneralizedCauchy5);
  JIZAI_CASE(CovGeneralizedCauchy7);
  JIZAI_CASE(CovGeneralizedCauchy9);
  JIZAI_CASE(CovSpherical);
  JIZAI_CASE(CovSpheroidal3);
  JIZAI_CASE(CovSpheroidal5);
  JIZAI_CASE(CovSpheroidal7);
  JIZAI_CASE(CovSpheroidal9);
  JIZAI_CASE(Triharmonic2D);
  JIZAI_CASE(Triharmonic3D);

#undef JIZAI_CASE

  throw std::runtime_error(std::format("unknown RBF name: '{}'", name));
}

}  // namespace jizai::rbf
