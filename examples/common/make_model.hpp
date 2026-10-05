#pragma once

#include <Eigen/Core>
#include <format>
#include <jizai/jizai.hpp>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "model_options.hpp"

inline bool is_identifier(const std::string& token) {
  static const std::regex re{"[A-Z_a-z][0-9A-Z_a-z]*"};

  return std::regex_match(token, re);
}

inline bool is_number(const std::string& token) {
  static const std::regex re{"-?([0-9]+\\.?|[0-9]*\\.[0-9]+)([Ee][+-]?[0-9]+)?"};

  return std::regex_match(token, re);
}

template <class Iterator>
void throw_unexpected_input(const Iterator& it, const Iterator& end) {
  if (it == end) {
    throw std::runtime_error("unexpected end of input");
  }

  throw std::runtime_error(std::format("unexpected token: '{}'", *it));
}

template <int Dim>
jizai::Model<Dim> make_model(const ModelOptions& opts) {
  using Mat = jizai::Mat<Dim>;
  using Model = jizai::Model<Dim>;
  using Rbf = jizai::rbf::Rbf<Dim>;

  std::vector<Rbf> rbfs;

  auto it = opts.rbf_args.begin();
  auto end = opts.rbf_args.end();
  while (it != end && is_identifier(*it)) {
    const auto& name = *it++;

    std::vector<double> params;
    while (it != end && is_number(*it)) {
      params.push_back(jizai::numeric::to_double(*it++));
    }

    auto rbf = jizai::rbf::make_rbf<Dim>(name, params);

    while (it != end) {
      if (*it == "aniso") {
        ++it;

        std::vector<double> aniso;
        for (auto i = 0; i < Dim * Dim; i++) {
          if (it != end && is_number(*it)) {
            aniso.push_back(jizai::numeric::to_double(*it++));
          } else {
            throw_unexpected_input(it, end);
          }
        }

        rbf.set_anisotropy(Eigen::Map<Mat>(aniso.data()));
      } else {
        break;
      }
    }

    rbfs.push_back(std::move(rbf));
  }

  if (it != end) {
    throw_unexpected_input(it, end);
  }

  Model m{std::move(rbfs), opts.poly_degree};
  m.set_nugget(opts.nugget);

  return m;
}
