#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <utility>

using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::write_table;
using polatory::common::valuesd;
using polatory::geometry::points3d;
using polatory::rbf::cov_exponential;

int main(int /*argc*/, char* argv[]) {
  try {
    points3d points = read_table(argv[1]);
    valuesd values = read_table(argv[2]);
    points3d prediction_points = read_table(argv[3]);

    double absolute_tolerance = 1e-4;

    cov_exponential<3> rbf({1.0, 0.02});

    auto poly_degree = 0;
    model<3> model(std::move(rbf), poly_degree);

    interpolant<3> interpolant(model);

    interpolant.fit(points, values, absolute_tolerance);
    auto prediction_values = interpolant.evaluate(prediction_points);

    write_table(argv[4], prediction_values);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
