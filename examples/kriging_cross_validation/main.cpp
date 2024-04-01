#include <Eigen/Core>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <polatory/kriging.hpp>
#include <polatory/polatory.hpp>
#include <tuple>
#include <utility>

#include "../common/make_model.hpp"
#include "parse_options.hpp"

using polatory::model;
using polatory::read_table;
using polatory::tabled;
using polatory::common::valuesd;
using polatory::geometry::pointsNd;
using polatory::kriging::cross_validate;
using polatory::point_cloud::distance_filter;

template <int Dim>
void main_impl(model<Dim>&& model, const options& opts) {
  using Points = pointsNd<Dim>;

  // Load points (x,y,z) and values (value).
  tabled table = read_table(opts.in_file);
  Points points = table(Eigen::all, Eigen::seqN(0, Dim));
  valuesd values = table.col(3);
  Eigen::VectorXi set_ids = table.col(4).cast<int>();

  // Remove very close points.
  std::tie(points, values) = distance_filter(points, opts.min_distance)(points, values);

  // Run the cross validation.
  auto residuals =
      cross_validate<Dim>(model, points, values, set_ids, opts.absolute_tolerance, opts.max_iter);

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
    switch (opts.dim) {
      case 1:
        main_impl(make_model<1>(opts.model_opts), opts);
        break;
      case 2:
        main_impl(make_model<2>(opts.model_opts), opts);
        break;
      case 3:
        main_impl(make_model<3>(opts.model_opts), opts);
        break;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
