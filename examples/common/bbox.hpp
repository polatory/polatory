#pragma once

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <polatory/polatory.hpp>
#include <string>
#include <vector>

namespace polatory::geometry {

inline void validate(boost::any& v, const std::vector<std::string>& values, bbox3d*, int) {
  namespace po = boost::program_options;

  if (values.size() != 6) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  v = bbox3d({numeric::to_double(values.at(0)), numeric::to_double(values.at(1)),
              numeric::to_double(values.at(2))},
             {numeric::to_double(values.at(3)), numeric::to_double(values.at(4)),
              numeric::to_double(values.at(5))});
}

}  // namespace polatory::geometry
