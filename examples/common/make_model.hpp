#pragma once

#include <polatory/polatory.hpp>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "model_options.hpp"

bool is_identifier(const std::string& token) {
  static const std::regex re{"[A-Z_a-z][0-9A-Z_a-z]*"};

  return std::regex_match(token, re);
}

bool is_number(const std::string& token) {
  static const std::regex re{"-?([0-9]+\\.?|[0-9]*\\.[0-9]+)([Ee][+-]?[0-9]+)?"};

  return std::regex_match(token, re);
}

template <int Dim>
polatory::model<Dim> make_model(const model_options& opts) {
  using Matrix = polatory::geometry::matrixNd<Dim>;
  using Model = polatory::model<Dim>;
  using RbfProxy = polatory::rbf::rbf_proxy<Dim>;

  std::vector<RbfProxy> rbfs;

  auto it = opts.rbf_args.begin();
  auto end = opts.rbf_args.end();
  while (it != end && is_identifier(*it)) {
    const auto& name = *it++;

    std::vector<double> params;
    while (it != end && is_number(*it)) {
      params.push_back(polatory::numeric::to_double(*it++));
    }

    auto rbf = polatory::rbf::make_rbf<Dim>(name, params);

    if (it != end && *it == "aniso") {
      ++it;

      std::vector<double> aniso_elems;
      while (it != end && is_number(*it)) {
        aniso_elems.push_back(polatory::numeric::to_double(*it++));
      }
      if (aniso_elems.size() != Dim * Dim) {
        throw std::runtime_error("wrong anisotropy size");
      }

      Matrix aniso;
      for (auto i = 0; i < Dim; ++i) {
        for (auto j = 0; j < Dim; ++j) {
          aniso(i, j) = aniso_elems.at(Dim * i + j);
        }
      }
      rbf.set_anisotropy(aniso);
    }

    rbfs.push_back(std::move(rbf));
  }

  if (it != end) {
    throw std::runtime_error("unexpected token in --rbf: " + *it);
  }

  Model m{std::move(rbfs), opts.poly_degree};
  m.set_nugget(opts.nugget);

  return m;
}
