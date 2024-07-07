#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <utility>

using polatory::interpolant;
using polatory::model;
using polatory::read_table;
using polatory::vectord;
using polatory::write_table;
using polatory::geometry::points3d;
using polatory::rbf::cov_exponential;

int main(int /*argc*/, char* argv[]) {
  try {
    points3d points = read_table(argv[1]);
    vectord values = read_table(argv[2]);
    points3d prediction_points = read_table(argv[3]);

    double tolerance = 1e-4;

    cov_exponential<3> rbf({1.0, 0.02});

    auto poly_degree = 0;
    model<3> model(std::move(rbf), poly_degree);

    interpolant<3> interpolant(model);

    interpolant.fit(points, values, tolerance);
    auto prediction_values = interpolant.evaluate(prediction_points);

    write_table(argv[4], prediction_values);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
