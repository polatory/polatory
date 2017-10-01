#ifndef FOMPPRIORITIES_HPP
#define FOMPPRIORITIES_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Utils/FEnv.hpp"

class FOmpPriorities{
    int insertionPositionP2M;
    int insertionPositionM2M;

    int insertionPositionP2MSend;
    int insertionPositionM2MSend;

    int insertionPositionM2L;
    int insertionPositionM2LExtern;
    int insertionPositionM2LLastLevel;
    int insertionPositionL2L;
    int insertionPositionL2P;
    int insertionPositionP2P;
    int insertionPositionP2PExtern;

    int treeHeight;

    int maxprio;

    bool inversePriorities;

    int scalePrio(const int inPrio) const{
        return (inversePriorities? maxprio-1-inPrio : inPrio);
    }

public:
    FOmpPriorities(const int inTreeHeight) :
        insertionPositionP2M(0), insertionPositionM2M(0), insertionPositionP2MSend(0),
        insertionPositionM2MSend(0), insertionPositionM2L(0), insertionPositionM2LExtern(0),
        insertionPositionM2LLastLevel(0), insertionPositionL2L(0), insertionPositionL2P(0), insertionPositionP2P(0),
        insertionPositionP2PExtern(0), treeHeight(inTreeHeight) , maxprio(0),
        inversePriorities(FEnv::GetBool("SCALFMM_INVERSE_PRIO", false)){
        if(inTreeHeight > 2){
            int incPrio = 0;

            FLOG( FLog::Controller << "Buckets:\n" );

            insertionPositionP2MSend = incPrio++;
            FLOG( FLog::Controller << "\t P2M Send "  << insertionPositionP2MSend << "\n" );
            insertionPositionP2M     = incPrio++;
            FLOG( FLog::Controller << "\t P2M "  << insertionPositionP2M << "\n" );

            insertionPositionM2MSend = incPrio++;
            FLOG( FLog::Controller << "\t M2M Send "  << insertionPositionM2MSend << "\n" );
            insertionPositionM2M     = incPrio++;
            FLOG( FLog::Controller << "\t M2M "  << insertionPositionM2M << "\n" );

            insertionPositionP2P       = incPrio++;
            FLOG( FLog::Controller << "\t P2P "  << insertionPositionP2P << "\n" );

            insertionPositionM2L     = incPrio++;
            FLOG( FLog::Controller << "\t M2L "  << insertionPositionM2L << "\n" );
            insertionPositionM2LExtern = incPrio++;
            FLOG( FLog::Controller << "\t M2L Outer "  << insertionPositionM2LExtern << "\n" );

            insertionPositionL2L     = incPrio++;
            FLOG( FLog::Controller << "\t L2L "  << insertionPositionL2L << "\n" );

            incPrio += (treeHeight-3) - 1;   // M2L is done treeHeight-2 times
            incPrio += (treeHeight-3) - 1;   // M2L is done treeHeight-2 times
            incPrio += (treeHeight-3) - 1;   // L2L is done treeHeight-3 times

            insertionPositionP2PExtern = incPrio++;
            FLOG( FLog::Controller << "\t P2P Outer "  << insertionPositionP2PExtern << "\n" );

            insertionPositionM2LLastLevel = incPrio++;
            FLOG( FLog::Controller << "\t M2L last "  << insertionPositionM2LLastLevel << "\n" );

            insertionPositionL2P     = incPrio++;
            FLOG( FLog::Controller << "\t L2P "  << insertionPositionL2P << "\n" );

            assert(incPrio == 8 + (treeHeight-3) + (treeHeight-3) + (treeHeight-3));
            maxprio = incPrio;
        }
        else{
            int incPrio = 0;

            insertionPositionP2MSend = -1;
            insertionPositionP2M     = -1;

            insertionPositionM2MSend = -1;
            insertionPositionM2M     = -1;

            insertionPositionM2L     = -1;
            insertionPositionM2LExtern = -1;
            insertionPositionM2LLastLevel = -1;

            insertionPositionL2L     = -1;

            insertionPositionP2P     = incPrio++;
            insertionPositionP2PExtern = insertionPositionP2P;

            insertionPositionL2P     = -1;
            assert(incPrio == 1);
            maxprio = incPrio;
        }

        if(inversePriorities){
            FLOG( FLog::Controller << "FOmpPriorities -- priorities are inversed\n" );
            FLOG( FLog::Controller << "FOmpPriorities -- the higher the more prioritized\n" );
        }
        else{
            FLOG( FLog::Controller << "FOmpPriorities -- priorities are made for heteroprio\n" );
            FLOG( FLog::Controller << "FOmpPriorities -- can be seen as the lower the more prioritized\n" );
        }
    }

    int getMaxPrio() const{
        return maxprio;
    }

    int getInsertionPosP2M() const {
        return scalePrio(insertionPositionP2M);
    }
    int getInsertionPosM2M(const int /*inLevel*/) const {
        return scalePrio(insertionPositionM2M);
    }
    int getInsertionPosM2L(const int inLevel) const {
        return scalePrio(inLevel==treeHeight-1? insertionPositionM2LLastLevel : insertionPositionM2L + (inLevel - 2)*3);
    }
    int getInsertionPosM2LExtern(const int inLevel) const {
        return scalePrio(inLevel==treeHeight-1? insertionPositionM2LLastLevel : insertionPositionM2LExtern + (inLevel - 2)*3);
    }
    int getInsertionPosL2L(const int inLevel) const {
        return scalePrio(insertionPositionL2L + (inLevel - 2)*3);
    }
    int getInsertionPosL2P() const {
        return scalePrio(insertionPositionL2P);
    }
    int getInsertionPosP2P() const {
        return scalePrio(insertionPositionP2P);
    }
    int getInsertionPosP2PExtern() const {
        return scalePrio(insertionPositionP2PExtern);
    }
};

#endif // FOMPPRIORITIES_HPP

