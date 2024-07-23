#pragma once

#include <polatory/common/io.hpp>
#include <polatory/fmm/interpolator_configuration.hpp>
#include <polatory/rbf/make_rbf.hpp>
#include <polatory/rbf/rbf_proxy.hpp>

namespace polatory::common {

template <int Dim>
struct Read<rbf::rbf_proxy<Dim>> {
  void operator()(std::istream& is, rbf::rbf_proxy<Dim>& t) const {
    using Matrix = geometry::matrixNd<Dim>;

    std::string short_name;
    read(is, short_name);
    std::vector<double> parameters;
    read(is, parameters);
    Matrix anisotropy;
    read(is, anisotropy);
    fmm::interpolator_configuration config;
    read(is, config);

    auto rbf = rbf::make_rbf<Dim>(short_name, parameters);
    rbf.set_anisotropy(anisotropy);
    rbf.set_interpolator_configuration(config);

    std::swap(t, rbf);
  }
};

template <int Dim>
struct Write<rbf::rbf_proxy<Dim>> {
  void operator()(std::ostream& os, const rbf::rbf_proxy<Dim>& t) const {
    write(os, t.short_name());
    write(os, t.parameters());
    write(os, t.anisotropy());
    write(os, t.interpolator_configuration());
  }
};

}  // namespace polatory::common
