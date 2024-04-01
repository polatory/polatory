#pragma once

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

struct rbf_options {
  std::string name;
  std::vector<double> params;
  std::vector<double> aniso;
};

rbf_options parse_rbf_options(const std::vector<std::string>& values) {
  namespace po = boost::program_options;

  if (values.size() < 1) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  auto it = values.begin();
  auto end = values.end();

  rbf_options opts;
  opts.name = *it++;

  while (it != end && *it != "aniso") {
    opts.params.push_back(polatory::numeric::to_double(*it++));
  }

  if (it != end) {
    it++;  // Skip "aniso".
    if (it == end) {
      throw po::validation_error(po::validation_error::invalid_option_value);
    }

    while (it != end) {
      opts.aniso.push_back(polatory::numeric::to_double(*it++));
    }
  }

  return opts;
}

inline void validate(boost::any& v, const std::vector<std::string>& values, rbf_options*, int) {
  v = parse_rbf_options(values);
}

inline void validate(boost::any& v, const std::vector<std::string>& values,
                     std::optional<rbf_options>*, int) {
  v = std::optional{parse_rbf_options(values)};
}

struct model_options {
  rbf_options rbf;
  std::optional<rbf_options> rbf2;
  double nugget{};
  int poly_degree{};
};

boost::program_options::options_description make_model_options_description(model_options& opts) {
  namespace po = boost::program_options;

  po::options_description opts_desc("Model", 80, 50);
  opts_desc.add_options()                                                               //
      ("rbf", po::value(&opts.rbf)->multitoken()->required()->value_name("..."),        //
       rbf_cov_list)                                                                    //
      ("rbf2", po::value(&opts.rbf2)->multitoken()->value_name("..."),                  //
       "The second structure")                                                          //
      ("nugget", po::value(&opts.nugget)->default_value(0.0, "0.")->value_name("VAL"),  //
       "Nugget of the model")                                                           //
      ("deg",
       po::value(&opts.poly_degree)
           ->default_value(polatory::model<1>::kMinRequiredPolyDegree, "AUTO")
           ->value_name("-1|0|1|2"),  //
       "Degree of the polynomial trend");

  return opts_desc;
}

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
