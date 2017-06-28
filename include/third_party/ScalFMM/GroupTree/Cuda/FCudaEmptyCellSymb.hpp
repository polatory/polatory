#ifndef FCUDAEMPTYCELLSYMB_HPP
#define FCUDAEMPTYCELLSYMB_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Containers/FTreeCoordinate.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

struct alignas(FStarPUDefaultAlign::StructAlign) FCudaEmptyCellSymb {
    MortonIndex mortonIndex;
    int coordinates[3];
};


#endif // FCUDAEMPTYCELLSYMB_HPP

