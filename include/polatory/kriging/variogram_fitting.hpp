#pragma once

#include <polatory/kriging/empirical_variogram.hpp>
#include <polatory/kriging/weight_function.hpp>
#include <polatory/model.hpp>
#include <vector>

namespace polatory::kriging {

class variogram_fitting {
 public:
  variogram_fitting(const empirical_variogram& emp_variog, const model& model,
                    const weight_function& weight_fn);

  const std::vector<double>& parameters() const;

 private:
  std::vector<double> params_;
};

}  // namespace polatory::kriging
