#ifndef FBASICCELLPOD_HPP
#define FBASICCELLPOD_HPP


#include "../../Utils/FGlobal.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

struct alignas(FStarPUDefaultAlign::StructAlign) FBasicCellPOD {
    MortonIndex mortonIndex;
    int coordinates[3];
    int level;
};

#endif // FBASICCELLPOD_HPP
