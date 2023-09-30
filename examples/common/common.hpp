#pragma once

#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <memory>
#include <polatory/polatory.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace Eigen {

inline void validate(boost::any& v, const std::vector<std::string>& values,
                     polatory::geometry::linear_transformation3d*, int) {
  namespace po = boost::program_options;

  if (values.size() != 9) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  polatory::geometry::linear_transformation3d aniso;
  aniso << boost::lexical_cast<double>(values[0]), boost::lexical_cast<double>(values[1]),
      boost::lexical_cast<double>(values[2]), boost::lexical_cast<double>(values[3]),
      boost::lexical_cast<double>(values[4]), boost::lexical_cast<double>(values[5]),
      boost::lexical_cast<double>(values[6]), boost::lexical_cast<double>(values[7]),
      boost::lexical_cast<double>(values[8]);

  v = aniso;
}

}  // namespace Eigen

namespace polatory {
namespace geometry {

inline void validate(boost::any& v, const std::vector<std::string>& values, bbox3d*, int) {
  namespace po = boost::program_options;

  if (values.size() != 6) {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }

  v = bbox3d({boost::lexical_cast<double>(values[0]), boost::lexical_cast<double>(values[1]),
              boost::lexical_cast<double>(values[2])},
             {boost::lexical_cast<double>(values[3]), boost::lexical_cast<double>(values[4]),
              boost::lexical_cast<double>(values[5])});
}

}  // namespace geometry
}  // namespace polatory

extern const char* const cov_list;
extern const char* const rbf_cov_list;
