#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <polatory/polatory.hpp>
#include <utility>

using polatory::Interpolant;
using polatory::kAll;
using polatory::MatX;
using polatory::Model;
using polatory::read_table;
using polatory::VecX;
using polatory::write_table;
using polatory::common::concatenate_cols;
using polatory::geometry::Points3;
using polatory::rbf::CovExponential;

int main(int /*argc*/, char* argv[]) {
  try {
    MatX table = read_table(argv[1]);
    Points3 points = table(kAll, {0, 1, 2});
    VecX values = table.col(3);
    Points3 prediction_points = read_table(argv[2]);

    auto tolerance = 1e-3;
    auto accuracy = 1e-5;

    CovExponential<3> rbf({1.0, 0.06});
    Model<3> model(std::move(rbf), -1);

    Interpolant<3> interpolant(model);

    auto fit_start = std::chrono::high_resolution_clock::now();
    interpolant.fit(points, values, tolerance, 100, accuracy);
    auto fit_end = std::chrono::high_resolution_clock::now();

    auto eval_start = std::chrono::high_resolution_clock::now();
    auto prediction_values = interpolant.evaluate(prediction_points, accuracy);
    auto eval_end = std::chrono::high_resolution_clock::now();

    auto fit_time =
        1e-3 *
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(fit_end - fit_start).count());
    auto eval_time =
        1e-3 *
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start).count());
    std::cout << std::fixed << std::setprecision(3)  //
              << "fitting took " << fit_time << "s" << std::endl
              << "evaluation took " << eval_time << "s" << std::endl
              << std::defaultfloat;

    write_table(argv[3], concatenate_cols<MatX>(prediction_points, prediction_values));

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
