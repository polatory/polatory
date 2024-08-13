#pragma once

#include <string>
#include <vector>

struct GlobalOptions {
  bool help{};
};

#define POLATORY_COMMAND(PREFIX, NAME)                                                       \
  class PREFIX##Command {                                                                    \
   public:                                                                                   \
    static inline const std::string kName = NAME;                                            \
                                                                                             \
    static void run(const std::vector<std::string>& args, const GlobalOptions& global_opts); \
  };

POLATORY_COMMAND(CreateModel, "create-model");
POLATORY_COMMAND(CrossValidate, "cross-validate");
POLATORY_COMMAND(EstimateNormals, "estimate-normals");
POLATORY_COMMAND(Evaluate, "evaluate");
POLATORY_COMMAND(ExtractModel, "extract-model");
POLATORY_COMMAND(Fit, "fit");
POLATORY_COMMAND(FitModelToVariogram, "fit-model-to-variogram");
POLATORY_COMMAND(Isosurface, "isosurface");
POLATORY_COMMAND(NormalsToSdf, "normals-to-sdf");
POLATORY_COMMAND(ShowModel, "show-model");
POLATORY_COMMAND(ShowVariogram, "show-variogram");
POLATORY_COMMAND(Surface25D, "surface-25d");
POLATORY_COMMAND(Unique, "unique");
POLATORY_COMMAND(Variogram, "variogram");
