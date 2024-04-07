#pragma once

#include <string>
#include <vector>

struct global_options {
  bool help{};
};

#define POLATORY_COMMAND(PREFIX, NAME)                                                        \
  class PREFIX##_command {                                                                    \
   public:                                                                                    \
    static inline const std::string kName = NAME;                                             \
                                                                                              \
    static void run(const std::vector<std::string>& args, const global_options& global_opts); \
  };

POLATORY_COMMAND(create_model, "create-model");
POLATORY_COMMAND(cross_validate, "cross-validate");
POLATORY_COMMAND(estimate_normals, "estimate-normals");
POLATORY_COMMAND(evaluate, "evaluate");
POLATORY_COMMAND(extract_model, "extract-model");
POLATORY_COMMAND(fit, "fit");
POLATORY_COMMAND(fit_model_to_variogram, "fit-model-to-variogram");
POLATORY_COMMAND(isosurface, "isosurface");
POLATORY_COMMAND(normals_to_sdf, "normals-to-sdf");
POLATORY_COMMAND(show_model, "show-model");
POLATORY_COMMAND(show_variogram, "show-variogram");
POLATORY_COMMAND(surface_25d, "surface-25d");
POLATORY_COMMAND(unique, "unique");
POLATORY_COMMAND(variogram, "variogram");
