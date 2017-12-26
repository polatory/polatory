// See LICENCE file at project root
#ifndef FCORECOMMON_HPP
#define FCORECOMMON_HPP

#include "../Utils/FGlobal.hpp"
#include "../Utils/FAssert.hpp"

#ifdef SCALFMM_USE_EZTRACE
extern "C" {
#include "eztrace.h"
}
#endif
/**
 * @brief The FFmmOperations enum
 * To chose which operation has to be performed.
 */
enum FFmmOperations {
    FFmmP2P   = (1 << 0),
    FFmmP2M  = (1 << 1),
    FFmmM2M = (1 << 2),
    FFmmM2L  = (1 << 3),
    FFmmL2L  = (1 << 4),
    FFmmL2P  = (1 << 5),
//
    FFmmNearField = FFmmP2P,
    FFmmFarField  = (FFmmP2M|FFmmM2M|FFmmM2L|FFmmL2L|FFmmL2P),
//
    FFmmNearAndFarFields = (FFmmNearField|FFmmFarField)
};

/**
 * \brief Base class of algorithms
 *
 * This class is an abstract algorithm to be able to use the FAlgorithmBuilder
 * and execute from an abstract pointer.
 */
class FAbstractAlgorithm {
protected:

    int upperWorkingLevel; ///< Where to start the work
    int lowerWorkingLevel; ///< Where to end the work (exclusive)
    int nbLevelsInTree;    ///< Height of the tree

    void setNbLevelsInTree(const int inNbLevelsInTree){
        nbLevelsInTree       = inNbLevelsInTree;
        lowerWorkingLevel = nbLevelsInTree;
    }

    void validateLevels() const {
        FAssertLF(FAbstractAlgorithm::upperWorkingLevel <= FAbstractAlgorithm::lowerWorkingLevel);
        FAssertLF(2 <= FAbstractAlgorithm::upperWorkingLevel);
    }

    virtual void executeCore(const unsigned operationsToProceed) = 0;

public:
    FAbstractAlgorithm()
        : upperWorkingLevel(2), lowerWorkingLevel(0), nbLevelsInTree(-1){
    }

    virtual ~FAbstractAlgorithm(){
    }

    /** \brief Execute the whole fmm for given levels. */
    virtual void execute(const int inUpperWorkingLevel, const int inLowerWorkingLevel) final {
        upperWorkingLevel = inUpperWorkingLevel;
        lowerWorkingLevel = inLowerWorkingLevel;
        validateLevels();
        executeCore(FFmmNearAndFarFields);
    }

    /** \brief Execute the whole fmm. */
    virtual void execute() final {
        upperWorkingLevel = 2;
        lowerWorkingLevel = nbLevelsInTree;
        validateLevels();
        executeCore(FFmmNearAndFarFields);
    }

    /** \brief Execute only some FMM operations for given levels. */
    virtual void execute(const unsigned operationsToProceed, const int inUpperWorkingLevel, const int inLowerWorkingLevel) final {
        upperWorkingLevel = inUpperWorkingLevel;
        lowerWorkingLevel = inLowerWorkingLevel;
        validateLevels();
        executeCore(operationsToProceed);
    }

    /** \brief Execute only some steps. */
    virtual void execute(const unsigned operationsToProceed) final {
        upperWorkingLevel = 2;
        lowerWorkingLevel = nbLevelsInTree;
        validateLevels();
        executeCore(operationsToProceed);
    }
};




#endif // FCORECOMMON_HPP
