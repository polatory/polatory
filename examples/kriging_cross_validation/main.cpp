#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <polatory/polatory.hpp>
#include <tuple>

#include "parse_options.hpp"

using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::kriging::k_fold_cross_validation;
using polatory::point_cloud::distance_filter;
using polatory::rbf::biharmonic2d;
using polatory::rbf::biharmonic3d;
using polatory::rbf::cov_exponential;
using polatory::rbf::cov_spheroidal3;
using polatory::rbf::cov_spheroidal5;
using polatory::rbf::cov_spheroidal7;
using polatory::rbf::cov_spheroidal9;
using polatory::rbf::multiquadric1;

template <class Rbf>
void main_impl(const options& opts) {
  using Model = model<Rbf>;

  // Load points (x,y,z) and values (value).
  tabled table = read_table(opts.in_file);
  points3d points = table(Eigen::all, {0, 1, 2});
  valuesd values = table.col(3);

  // Remove very close points.
  std::tie(points, values) = distance_filter(points, opts.min_distance)(points, values);

  // Define the model.
  Rbf rbf(opts.rbf_params);
  rbf.set_anisotropy(opts.aniso);
  Model model(rbf, opts.poly_dimension, opts.poly_degree);
  model.set_nugget(opts.nugget);

  // Run the cross validation.
  auto residuals = k_fold_cross_validation(model, points, values, opts.absolute_tolerance,
                                           opts.max_iter, opts.k);

  std::cout << "Estimated mean absolute error: " << std::endl
            << std::setw(12) << residuals.template lpNorm<1>() / static_cast<double>(points.rows())
            << std::endl
            << "Estimated root mean square error: " << std::endl
            << std::setw(12)
            << std::sqrt(residuals.squaredNorm() / static_cast<double>(points.rows())) << std::endl;
}

int main(int argc, const char* argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    if (opts.rbf_name == "bh2") {
      main_impl<biharmonic2d<3>>(opts);
    } else if (opts.rbf_name == "bh3") {
      main_impl<biharmonic3d<3>>(opts);
    } else if (opts.rbf_name == "exp") {
      main_impl<cov_exponential<3>>(opts);
    } else if (opts.rbf_name == "sp3") {
      main_impl<cov_spheroidal3<3>>(opts);
    } else if (opts.rbf_name == "sp5") {
      main_impl<cov_spheroidal5<3>>(opts);
    } else if (opts.rbf_name == "sp7") {
      main_impl<cov_spheroidal7<3>>(opts);
    } else if (opts.rbf_name == "sp9") {
      main_impl<cov_spheroidal9<3>>(opts);
    } else if (opts.rbf_name == "mq1") {
      main_impl<multiquadric1<3>>(opts);
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
