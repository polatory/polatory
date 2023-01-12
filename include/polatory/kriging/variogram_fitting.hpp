#pragma once

#include <vector>

#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/kriging/weight_function.hpp>
#include <polatory/model.hpp>

namespace polatory {
namespace kriging {

class variogram_fitting {
public:
  variogram_fitting(const empirical_variogram& emp_variog, const model& model,
    const weight_function& weight_fn);

  const std::vector<double>& parameters() const;

private:
  std::vector<double> params_;
};

}  // namespace kriging
}  // namespace polatory
