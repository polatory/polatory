#pragma once

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <jizai/jizai.hpp>
#include <string>
#include <vector>

namespace jizai::geometry {

inline void validate(boost::any& v, const std::vector<std::string>& values, Bbox3*, int) {
  namespace po = boost::program_options;

  if (values.size() != 6) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  v = Bbox3({numeric::to_double(values.at(0)), numeric::to_double(values.at(1)),
             numeric::to_double(values.at(2))},
            {numeric::to_double(values.at(3)), numeric::to_double(values.at(4)),
             numeric::to_double(values.at(5))});
}

}  // namespace jizai::geometry
