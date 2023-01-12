#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <tuple>

#include <polatory/polatory.hpp>

#include "parse_options.hpp"

using polatory::common::take_cols;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::kriging::k_fold_cross_validation;
using polatory::model;
using polatory::point_cloud::distance_filter;
using polatory::read_table;
using polatory::tabled;

int main(int argc, const char *argv[]) {
  try {
    auto opts = parse_options(argc, argv);

    // Load points (x,y,z) and values (value).
    tabled table = read_table(opts.in_file);
    points3d points = take_cols(table, 0, 1, 2);
    valuesd values = table.col(3);

    // Remove very close points.
    std::tie(points, values) = distance_filter(points, opts.min_distance)
      .filtered(points, values);

    // Define the model.
    auto rbf = make_rbf(opts.rbf_name, opts.rbf_params);
    rbf->set_anisotropy(opts.aniso);
    model model(*rbf, opts.poly_dimension, opts.poly_degree);
    model.set_nugget(opts.nugget);

    // Run the cross validation.
    auto residuals = k_fold_cross_validation(model, points, values, opts.absolute_tolerance, opts.k);

    std::cout << "Estimated mean absolute error: " << std::endl
              << std::setw(12) << residuals.lpNorm<1>() / points.rows() << std::endl
              << "Estimated root mean square error: " << std::endl
              << std::setw(12) << std::sqrt(residuals.squaredNorm() / points.rows()) << std::endl;

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
