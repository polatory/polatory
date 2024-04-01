#pragma once

#include <polatory/polatory.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "model_options.hpp"

template <int Dim>
polatory::rbf::rbf_proxy<Dim> make_rbf(const rbf_options& opts) {
  auto rbf = polatory::rbf::make_rbf<Dim>(opts.name, opts.params);

  if (opts.aniso.size() != 0) {
    if (opts.aniso.size() != Dim * Dim) {
      throw std::runtime_error("wrong anisotropy size");
    }
    polatory::geometry::matrixNd<Dim> aniso;
    for (int i = 0; i < Dim; ++i) {
      for (int j = 0; j < Dim; ++j) {
        aniso(i, j) = opts.aniso.at(Dim * i + j);
      }
    }
    rbf.set_anisotropy(aniso);
  }

  return rbf;
}

template <int Dim>
polatory::model<Dim> make_model(const model_options& opts) {
  using Model = polatory::model<Dim>;
  using RbfProxy = polatory::rbf::rbf_proxy<Dim>;

  std::vector<RbfProxy> rbfs;
  rbfs.push_back(make_rbf<Dim>(opts.rbf));

  if (opts.rbf2) {
    rbfs.push_back(make_rbf<Dim>(*opts.rbf2));
  }

  Model m{std::move(rbfs), opts.poly_degree};
  m.set_nugget(opts.nugget);

  return m;
}
