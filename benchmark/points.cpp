#include <exception>
#include <iostream>
#include <polatory/geometry/sphere3d.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polatory.hpp>
#include <string>

using polatory::write_table;
using polatory::geometry::sphere3d;
using polatory::point_cloud::distance_filter;
using polatory::point_cloud::random_points;

int main(int /*argc*/, char* argv[]) {
  try {
    auto n_points = std::stoi(argv[1]);
    auto seed = std::stoi(argv[2]);

    auto points = random_points(sphere3d(), n_points, seed);
    points = distance_filter(points).filter(1e-8)(points);

    write_table(argv[3], points);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
