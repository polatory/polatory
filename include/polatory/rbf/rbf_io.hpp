#pragma once

#include <polatory/common/io.hpp>
#include <polatory/rbf/make_rbf.hpp>
#include <polatory/rbf/rbf.hpp>

namespace polatory::common {

template <int Dim>
struct Read<rbf::Rbf<Dim>> {
  void operator()(std::istream& is, rbf::Rbf<Dim>& t) const {
    using Mat = Mat<Dim>;

    std::string short_name;
    read(is, short_name);
    std::vector<double> parameters;
    read(is, parameters);
    Mat anisotropy;
    read(is, anisotropy);

    auto rbf = rbf::make_rbf<Dim>(short_name, parameters);
    rbf.set_anisotropy(anisotropy);

    std::swap(t, rbf);
  }
};

template <int Dim>
struct Write<rbf::Rbf<Dim>> {
  void operator()(std::ostream& os, const rbf::Rbf<Dim>& t) const {
    write(os, t.short_name());
    write(os, t.parameters());
    write(os, t.anisotropy());
  }
};

}  // namespace polatory::common
