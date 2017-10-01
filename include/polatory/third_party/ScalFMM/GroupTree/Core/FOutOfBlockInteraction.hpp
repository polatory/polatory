#ifndef FOUTOFBLOCKINTERACTION_HPP
#define FOUTOFBLOCKINTERACTION_HPP

#include "../../Utils/FGlobal.hpp"

#include "../StarPUUtils/FStarPUDefaultAlign.hpp"

struct  alignas(FStarPUDefaultAlign::StructAlign) OutOfBlockInteraction{
    MortonIndex outIndex;
    MortonIndex insideIndex;
    int relativeOutPosition;
    int insideIdxInBlock;
    int outsideIdxInBlock;
    // To sort
    bool operator <=(const OutOfBlockInteraction& other) const{
        return outIndex <= other.outIndex;
    }
};

#endif // FOUTOFBLOCKINTERACTION_HPP

