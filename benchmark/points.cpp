#include <exception>
#include <iostream>
#include <polatory/geometry/sphere3d.hpp>
#include <polatory/point_cloud/random_points.hpp>
#include <polatory/polatory.hpp>
#include <string>

using polatory::write_table;
using polatory::geometry::Sphere3;
using polatory::point_cloud::DistanceFilter;
using polatory::point_cloud::random_points;

int main(int /*argc*/, char* argv[]) {
  try {
    auto n_points = std::stoi(argv[1]);
    auto seed = std::stoi(argv[2]);

    auto points = random_points(Sphere3(), n_points, seed);
    points = DistanceFilter(points).filter(1e-8)(points);

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
