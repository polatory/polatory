#include <exception>
#include <iostream>
#include <polatory/polatory.hpp>
#include <utility>

using polatory::Interpolant;
using polatory::Model;
using polatory::read_table;
using polatory::VecX;
using polatory::write_table;
using polatory::geometry::Points3;
using polatory::rbf::CovExponential;

int main(int /*argc*/, char* argv[]) {
  try {
    Points3 points = read_table(argv[1]);
    VecX values = read_table(argv[2]);
    Points3 prediction_points = read_table(argv[3]);

    double tolerance = 1e-4;

    CovExponential<3> rbf({1.0, 0.02});

    auto poly_degree = 0;
    Model<3> model(std::move(rbf), poly_degree);

    Interpolant<3> interpolant(model);

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
