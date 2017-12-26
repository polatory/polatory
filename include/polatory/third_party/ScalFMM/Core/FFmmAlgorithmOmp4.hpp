#ifndef FFMMALGORITHMOMP4_HPP
#define FFMMALGORITHMOMP4_HPP

#include <omp.h>

#include "../Utils/FGlobal.hpp"
#include "../Utils/FAssert.hpp"
#include "../Utils/FLog.hpp"

#include "../Utils/FTic.hpp"

#include "../Containers/FOctree.hpp"
#include "../Containers/FVector.hpp"
#include "../Utils/FAlgorithmTimers.hpp"
#include "../Utils/FEnv.hpp"

#include "FCoreCommon.hpp"
#include "FP2PExclusion.hpp"

#undef commute_if_supported
#ifdef OPENMP_SUPPORT_COMMUTE
#define commute_if_supported commute
#else
#define commute_if_supported inout
#endif

#undef priority_if_supported
#ifdef OPENMP_SUPPORT_PRIORITY
#define priority_if_supported(x) priority(x)

enum FFmmAlgorithmOmp4_Priorities{
    FFmmAlgorithmOmp4_Prio_P2M = 9,
    FFmmAlgorithmOmp4_Prio_M2M = 8,
    FFmmAlgorithmOmp4_Prio_M2L_High = 7,
    FFmmAlgorithmOmp4_Prio_L2L = 6,
    FFmmAlgorithmOmp4_Prio_P2P_Big = 5,
    FFmmAlgorithmOmp4_Prio_M2L = 4,
    FFmmAlgorithmOmp4_Prio_L2P = 3,
    FFmmAlgorithmOmp4_Prio_P2P_Small = 2
};

#else
#define priority_if_supported(x)
#endif



/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class FFmmAlgorithmOmp4
 * @brief
 * Please read the license
 *
 * This class is a basic FMM algorithm
 * It just iterates on a tree and call the kernels with good arguments.
 *
 * Of course this class does not deallocate pointer given in arguements.
 */
template<class OctreeClass, class CellClass, class ContainerClass, class KernelClass, class LeafClass, class P2PExclusionClass = FP2PMiddleExclusion>
class FFmmAlgorithmOmp4 : public FAbstractAlgorithm, public FAlgorithmTimers {

    static_assert(sizeof(CellClass) > 1, "CellClass should be greater than one byte to ensure dependency coherency");


    template <int BlockSize = 256>
    class NoDeps{
    protected:
        std::list< std::array<char, BlockSize> > blocks;
        int currentIndex;

    public:
        NoDeps(const NoDeps&) = delete;
        NoDeps& operator=(const NoDeps&) = delete;

        NoDeps() : currentIndex(0){
            blocks.emplace_back();
        }

        template <class PtrType>
        PtrType getNextDep(){
            if(currentIndex == BlockSize){
                blocks.emplace_back();
                currentIndex = 0;
            }
            return reinterpret_cast<PtrType>(&blocks.back()[currentIndex++]);
        }

        void clear(){
            currentIndex = 0;
            blocks.clear();
            blocks.emplace_back();
        }
    };



    NoDeps<> nodeps;

    OctreeClass* const tree;       //< The octree to work on
    KernelClass** kernels;    //< The kernels

    const int MaxThreads;

    const int OctreeHeight;

    const int leafLevelSeparationCriteria;

    // Used with OPENMP_SUPPORT_PRIORITY
    size_t p2pPrioCriteria;

public:
    /** The constructor need the octree and the kernels used for computation
     * @param inTree the octree to work on
     * @param inKernels the kernels to call
     * An assert is launched if one of the arguments is null
     */
    FFmmAlgorithmOmp4(OctreeClass* const inTree, KernelClass* const inKernels, const int inLeafLevelSeperationCriteria = 1)
: tree(inTree) , kernels(nullptr),
  MaxThreads(FEnv::GetValue("SCALFMM_ALGO_NUM_THREADS",omp_get_max_threads())), OctreeHeight(tree->getHeight()), leafLevelSeparationCriteria(inLeafLevelSeperationCriteria),
      p2pPrioCriteria(0)
{

        FAssertLF(tree, "tree cannot be null");
        FAssertLF(inKernels, "kernels cannot be null");
        FAssertLF(leafLevelSeparationCriteria < 3, "Separation criteria should be < 3");

        this->kernels = new KernelClass*[MaxThreads];
        #pragma omp parallel num_threads(MaxThreads)
        {
            #pragma omp critical (InitFFmmAlgorithmOmp4)
            {
                this->kernels[omp_get_thread_num()] = new KernelClass(*inKernels);
            }
        }

        FAbstractAlgorithm::setNbLevelsInTree(tree->getHeight());

        FLOG(FLog::Controller << "FFmmAlgorithmOmp4 (Max Thread " << omp_get_max_threads() << ")\n");

        FAssertLF(KernelClass::NeedFinishedM2LEvent() == false, "FFmmAlgorithmOmp4 cannot notify for M2L at level ending.");

#ifdef OPENMP_SUPPORT_PRIORITY
        size_t nbLeaves = 0;
        size_t nbParticles = 0;

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();
        // for each leafs
        do{
            ContainerClass* taskParticlesTgt = octreeIterator.getCurrentListTargets();
            nbParticles += taskParticlesTgt->getNbParticles();
            nbLeaves    += 1;
        } while(octreeIterator.moveRight());
        p2pPrioCriteria = (nbParticles/nbLeaves);
#endif
}

    /** Default destructor */
    virtual ~FFmmAlgorithmOmp4(){
        for(int idxThread = 0 ; idxThread < MaxThreads ; ++idxThread){
            delete this->kernels[idxThread];
        }
        delete [] this->kernels;
    }

protected:

    const unsigned char* getCellMultipoleDepPtr(const CellClass* taskCell) const {
        return reinterpret_cast<const unsigned char*>(taskCell);
    }

    const unsigned char* getCellLocalDepPtr(const CellClass* taskCell) const {
        return reinterpret_cast<const unsigned char*>(taskCell) + 1;
    }


    /**
     * To execute the fmm algorithm
     * Call this function to run the complete algorithm
     */
    void executeCore(const unsigned operationsToProceed) override {

        #pragma omp parallel num_threads(MaxThreads)
        {
            #pragma omp master
            {
                Timers[NearTimer].tic();
                if( operationsToProceed & FFmmP2P )
                    directPass();
                Timers[NearTimer].tac();

                Timers[P2MTimer].tic();
                if(operationsToProceed & FFmmP2M)
                    bottomPass();
                Timers[P2MTimer].tac();

                Timers[M2MTimer].tic();
                if(operationsToProceed & FFmmM2M)
                    upwardPass();
                Timers[M2MTimer].tac();

                Timers[M2LTimer].tic();
                if(operationsToProceed & FFmmM2L)
                    transferPass();
                Timers[M2LTimer].tac();

                Timers[L2LTimer].tic();
                if(operationsToProceed & FFmmL2L)
                    downardPass();
                Timers[L2LTimer].tac();

                Timers[L2PTimer].tic();
                if( operationsToProceed & FFmmL2P)
                    mergePass();
                Timers[L2PTimer].tac();

                #pragma omp taskwait
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // P2M
    /////////////////////////////////////////////////////////////////////////////

    /** P2M */
    void bottomPass(){
        FLOG( FLog::Controller.write("\tStart Bottom Pass\n").write(FLog::Flush) );
        FLOG(FTic counterTime);

        typename OctreeClass::Iterator octreeIterator(tree);

        // Iterate on leafs
        octreeIterator.gotoBottomLeft();
        do{
            // We need the current cell that represent the leaf
            // and the list of particles
            CellClass* taskCell = octreeIterator.getCurrentCell();
            const unsigned char* taskCellDep = getCellMultipoleDepPtr(taskCell);
            ContainerClass* taskParticles = octreeIterator.getCurrentListSrc();
            #pragma omp task firstprivate(taskCell, taskCellDep, taskParticles) depend(inout:taskCellDep[0]) depend(in:taskParticles[0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_P2M)
            {
                kernels[omp_get_thread_num()]->P2M( taskCell , taskParticles);
            }
        } while(octreeIterator.moveRight());


        FLOG( FLog::Controller << "\tFinished (@Bottom Pass (P2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

    /////////////////////////////////////////////////////////////////////////////
    // Upward
    /////////////////////////////////////////////////////////////////////////////

    /** M2M */
    void upwardPass(){
        FLOG( FLog::Controller.write("\tStart Upward Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

        // Start from leal level - 1
        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();
        octreeIterator.moveUp();

        for(int idxLevel = OctreeHeight - 2 ; idxLevel > FAbstractAlgorithm::lowerWorkingLevel-1 ; --idxLevel){
            octreeIterator.moveUp();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        // for each levels
        for(int idxLevel = FMath::Min(OctreeHeight - 2, FAbstractAlgorithm::lowerWorkingLevel - 1) ; idxLevel >= FAbstractAlgorithm::upperWorkingLevel ; --idxLevel ){
            FLOG(FTic counterTimeLevel);
            // for each cells
            do{
                // We need the current cell and the child
                // child is an array (of 8 child) that may be null
                CellClass* taskCell = octreeIterator.getCurrentCell();
                CellClass* taskChild[8];
                memcpy(taskChild, octreeIterator.getCurrentChild(), 8*sizeof(CellClass*));

                const unsigned char* taskCellDep = getCellMultipoleDepPtr(taskCell);
                const unsigned char* taskChildMultipole[8] = {nullptr};
                int counterFill = 0;
                for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                    if(taskChild[idxChild]){
                        taskChildMultipole[counterFill++] = getCellMultipoleDepPtr(taskChild[idxChild]);
                    }
                }

                switch(counterFill){
                case 1:
                    #pragma omp task firstprivate(taskCell, taskCellDep, taskChild, taskChildMultipole, idxLevel) depend(inout:taskCellDep[0]) depend(in:taskChildMultipole[0][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_M2M)
                    {
                        kernels[omp_get_thread_num()]->M2M( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 2:
                    #pragma omp task firstprivate(taskCell, taskCellDep, taskChild, taskChildMultipole, idxLevel) depend(inout:taskCellDep[0]) depend(in:taskChildMultipole[0][0],taskChildMultipole[1][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_M2M)
                    {
                        kernels[omp_get_thread_num()]->M2M( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 3:
                    #pragma omp task firstprivate(taskCell, taskCellDep, taskChild, taskChildMultipole, idxLevel) depend(inout:taskCellDep[0]) depend(in:taskChildMultipole[0][0],taskChildMultipole[1][0],taskChildMultipole[2][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_M2M)
                    {
                        kernels[omp_get_thread_num()]->M2M( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 4:
                    #pragma omp task firstprivate(taskCell, taskCellDep, taskChild, taskChildMultipole, idxLevel) depend(inout:taskCellDep[0]) depend(in:taskChildMultipole[0][0],taskChildMultipole[1][0],taskChildMultipole[2][0],taskChildMultipole[3][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_M2M)
                    {
                        kernels[omp_get_thread_num()]->M2M( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 5:
                    #pragma omp task firstprivate(taskCell, taskCellDep, taskChild, taskChildMultipole, idxLevel) depend(inout:taskCellDep[0]) depend(in:taskChildMultipole[0][0],taskChildMultipole[1][0],taskChildMultipole[2][0],taskChildMultipole[3][0],taskChildMultipole[4][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_M2M)
                    {
                        kernels[omp_get_thread_num()]->M2M( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 6:
                    #pragma omp task firstprivate(taskCell, taskCellDep, taskChild, taskChildMultipole, idxLevel) depend(inout:taskCellDep[0]) depend(in:taskChildMultipole[0][0],taskChildMultipole[1][0],taskChildMultipole[2][0],taskChildMultipole[3][0],taskChildMultipole[4][0],taskChildMultipole[5][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_M2M)
                    {
                        kernels[omp_get_thread_num()]->M2M( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 7:
                    #pragma omp task firstprivate(taskCell, taskCellDep, taskChild, taskChildMultipole, idxLevel) depend(inout:taskCellDep[0]) depend(in:taskChildMultipole[0][0],taskChildMultipole[1][0],taskChildMultipole[2][0],taskChildMultipole[3][0],taskChildMultipole[4][0],taskChildMultipole[5][0],taskChildMultipole[6][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_M2M)
                    {
                        kernels[omp_get_thread_num()]->M2M( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 8:
                    #pragma omp task firstprivate(taskCell, taskCellDep, taskChild, taskChildMultipole, idxLevel) depend(inout:taskCellDep[0]) depend(in:taskChildMultipole[0][0],taskChildMultipole[1][0],taskChildMultipole[2][0],taskChildMultipole[3][0],taskChildMultipole[4][0],taskChildMultipole[5][0],taskChildMultipole[6][0],taskChildMultipole[7][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_M2M)
                    {
                        kernels[omp_get_thread_num()]->M2M( taskCell , taskChild, idxLevel);
                    }
                    break;
                default:
                    FAssertLF(0, "Thus must not be possible");
                }
            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveUp();
            octreeIterator = avoidGotoLeftIterator;// equal octreeIterator.moveUp(); octreeIterator.gotoLeft();

            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }


        FLOG( FLog::Controller << "\tFinished (@Upward Pass (M2M) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

    /////////////////////////////////////////////////////////////////////////////
    // Transfer
    /////////////////////////////////////////////////////////////////////////////

    /** M2L  */
    void transferPass(){
      #ifdef SCALFMM_USE_EZTRACE

      eztrace_start();
#endif
        this->transferPassWithOutFinalize() ;
#ifdef SCALFMM_USE_EZTRACE
      eztrace_stop();
#endif
        }

    void transferPassWithOutFinalize(){
        FLOG( FLog::Controller.write("\tStart Downward Pass (M2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

        typename OctreeClass::Iterator octreeIterator(tree);
        // Goto the right level
        octreeIterator.moveDown();
        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }
        ////////////////////////////////////////////////////////////////
        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);
        //
        // for each levels
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < FAbstractAlgorithm::lowerWorkingLevel ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);
            const int separationCriteria = (idxLevel != FAbstractAlgorithm::lowerWorkingLevel-1 ? 1 : leafLevelSeparationCriteria);
            // for each cell we apply the M2L with all cells in the implicit interaction list
            do{
                const CellClass* taskNeigh[342];
                int neighborPositions[342];
                const int counter = tree->getInteractionNeighbors(taskNeigh, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(), idxLevel, separationCriteria);
                if(counter){
                    CellClass* taskCell = octreeIterator.getCurrentCell();
                    const unsigned char* taskCellLocal = getCellLocalDepPtr(taskCell);

                    for(int idxNoDep = counter ; idxNoDep < 342 ; ++idxNoDep){
                        taskNeigh[idxNoDep] = nodeps.template getNextDep<CellClass*>();
                    }

                    #pragma omp task firstprivate(taskCell,taskCellLocal, taskNeigh, neighborPositions, idxLevel, counter) depend(commute_if_supported:taskCellLocal[0]) depend(in:taskNeigh[0][0] , taskNeigh[1][0] , taskNeigh[2][0] , taskNeigh[3][0] , taskNeigh[4][0] , taskNeigh[5][0] , taskNeigh[6][0] , taskNeigh[7][0] , taskNeigh[8][0] , taskNeigh[9][0] , taskNeigh[10][0] , taskNeigh[11][0] , taskNeigh[12][0] , taskNeigh[13][0] , taskNeigh[14][0] , taskNeigh[15][0] , taskNeigh[16][0] , taskNeigh[17][0] , taskNeigh[18][0] , taskNeigh[19][0] , taskNeigh[20][0] , taskNeigh[21][0] , taskNeigh[22][0] , taskNeigh[23][0] , taskNeigh[24][0] , taskNeigh[25][0] , taskNeigh[26][0] , taskNeigh[27][0] , taskNeigh[28][0] , taskNeigh[29][0] , taskNeigh[30][0] , taskNeigh[31][0] , taskNeigh[32][0] , taskNeigh[33][0] , taskNeigh[34][0] , taskNeigh[35][0] , taskNeigh[36][0] , taskNeigh[37][0] , taskNeigh[38][0] , taskNeigh[39][0] , taskNeigh[40][0] , taskNeigh[41][0] , taskNeigh[42][0] , taskNeigh[43][0] , taskNeigh[44][0] , taskNeigh[45][0] , taskNeigh[46][0] , taskNeigh[47][0] , taskNeigh[48][0] , taskNeigh[49][0] , taskNeigh[50][0] , taskNeigh[51][0] , taskNeigh[52][0] , taskNeigh[53][0] , taskNeigh[54][0] , taskNeigh[55][0] , taskNeigh[56][0] , taskNeigh[57][0] , taskNeigh[58][0] , taskNeigh[59][0] , taskNeigh[60][0] , taskNeigh[61][0] , taskNeigh[62][0] , taskNeigh[63][0] , taskNeigh[64][0] , taskNeigh[65][0] , taskNeigh[66][0] , taskNeigh[67][0] , taskNeigh[68][0] , taskNeigh[69][0] , taskNeigh[70][0] , taskNeigh[71][0] , taskNeigh[72][0] , taskNeigh[73][0] , taskNeigh[74][0] , taskNeigh[75][0] , taskNeigh[76][0] , taskNeigh[77][0] , taskNeigh[78][0] , taskNeigh[79][0] , taskNeigh[80][0] , taskNeigh[81][0] , taskNeigh[82][0] , taskNeigh[83][0] , taskNeigh[84][0] , taskNeigh[85][0] , taskNeigh[86][0] , taskNeigh[87][0] , taskNeigh[88][0] , taskNeigh[89][0] , taskNeigh[90][0] , taskNeigh[91][0] , taskNeigh[92][0] , taskNeigh[93][0] , taskNeigh[94][0] , taskNeigh[95][0] , taskNeigh[96][0] , taskNeigh[97][0] , taskNeigh[98][0] , taskNeigh[99][0] , taskNeigh[100][0] , taskNeigh[101][0] , taskNeigh[102][0] , taskNeigh[103][0] , taskNeigh[104][0] , taskNeigh[105][0] , taskNeigh[106][0] , taskNeigh[107][0] , taskNeigh[108][0] , taskNeigh[109][0] , taskNeigh[110][0] , taskNeigh[111][0] , taskNeigh[112][0] , taskNeigh[113][0] , taskNeigh[114][0] , taskNeigh[115][0] , taskNeigh[116][0] , taskNeigh[117][0] , taskNeigh[118][0] , taskNeigh[119][0] , taskNeigh[120][0] , taskNeigh[121][0] , taskNeigh[122][0] , taskNeigh[123][0] , taskNeigh[124][0] , taskNeigh[125][0] , taskNeigh[126][0] , taskNeigh[127][0] , taskNeigh[128][0] , taskNeigh[129][0] , taskNeigh[130][0] , taskNeigh[131][0] , taskNeigh[132][0] , taskNeigh[133][0] , taskNeigh[134][0] , taskNeigh[135][0] , taskNeigh[136][0] , taskNeigh[137][0] , taskNeigh[138][0] , taskNeigh[139][0] , taskNeigh[140][0] , taskNeigh[141][0] , taskNeigh[142][0] , taskNeigh[143][0] , taskNeigh[144][0] , taskNeigh[145][0] , taskNeigh[146][0] , taskNeigh[147][0] , taskNeigh[148][0] , taskNeigh[149][0] , taskNeigh[150][0] , taskNeigh[151][0] , taskNeigh[152][0] , taskNeigh[153][0] , taskNeigh[154][0] , taskNeigh[155][0] , taskNeigh[156][0] , taskNeigh[157][0] , taskNeigh[158][0] , taskNeigh[159][0] , taskNeigh[160][0] , taskNeigh[161][0] , taskNeigh[162][0] , taskNeigh[163][0] , taskNeigh[164][0] , taskNeigh[165][0] , taskNeigh[166][0] , taskNeigh[167][0] , taskNeigh[168][0] , taskNeigh[169][0] , taskNeigh[170][0] , taskNeigh[171][0] , taskNeigh[172][0] , taskNeigh[173][0] , taskNeigh[174][0] , taskNeigh[175][0] , taskNeigh[176][0] , taskNeigh[177][0] , taskNeigh[178][0] , taskNeigh[179][0] , taskNeigh[180][0] , taskNeigh[181][0] , taskNeigh[182][0] , taskNeigh[183][0] , taskNeigh[184][0] , taskNeigh[185][0] , taskNeigh[186][0] , taskNeigh[187][0] , taskNeigh[188][0] , taskNeigh[189][0] , taskNeigh[190][0] , taskNeigh[191][0] , taskNeigh[192][0] , taskNeigh[193][0] , taskNeigh[194][0] , taskNeigh[195][0] , taskNeigh[196][0] , taskNeigh[197][0] , taskNeigh[198][0] , taskNeigh[199][0] , taskNeigh[200][0] , taskNeigh[201][0] , taskNeigh[202][0] , taskNeigh[203][0] , taskNeigh[204][0] , taskNeigh[205][0] , taskNeigh[206][0] , taskNeigh[207][0] , taskNeigh[208][0] , taskNeigh[209][0] , taskNeigh[210][0] , taskNeigh[211][0] , taskNeigh[212][0] , taskNeigh[213][0] , taskNeigh[214][0] , taskNeigh[215][0] , taskNeigh[216][0] , taskNeigh[217][0] , taskNeigh[218][0] , taskNeigh[219][0] , taskNeigh[220][0] , taskNeigh[221][0] , taskNeigh[222][0] , taskNeigh[223][0] , taskNeigh[224][0] , taskNeigh[225][0] , taskNeigh[226][0] , taskNeigh[227][0] , taskNeigh[228][0] , taskNeigh[229][0] , taskNeigh[230][0] , taskNeigh[231][0] , taskNeigh[232][0] , taskNeigh[233][0] , taskNeigh[234][0] , taskNeigh[235][0] , taskNeigh[236][0] , taskNeigh[237][0] , taskNeigh[238][0] , taskNeigh[239][0] , taskNeigh[240][0] , taskNeigh[241][0] , taskNeigh[242][0] , taskNeigh[243][0] , taskNeigh[244][0] , taskNeigh[245][0] , taskNeigh[246][0] , taskNeigh[247][0] , taskNeigh[248][0] , taskNeigh[249][0] , taskNeigh[250][0] , taskNeigh[251][0] , taskNeigh[252][0] , taskNeigh[253][0] , taskNeigh[254][0] , taskNeigh[255][0] , taskNeigh[256][0] , taskNeigh[257][0] , taskNeigh[258][0] , taskNeigh[259][0] , taskNeigh[260][0] , taskNeigh[261][0] , taskNeigh[262][0] , taskNeigh[263][0] , taskNeigh[264][0] , taskNeigh[265][0] , taskNeigh[266][0] , taskNeigh[267][0] , taskNeigh[268][0] , taskNeigh[269][0] , taskNeigh[270][0] , taskNeigh[271][0] , taskNeigh[272][0] , taskNeigh[273][0] , taskNeigh[274][0] , taskNeigh[275][0] , taskNeigh[276][0] , taskNeigh[277][0] , taskNeigh[278][0] , taskNeigh[279][0] , taskNeigh[280][0] , taskNeigh[281][0] , taskNeigh[282][0] , taskNeigh[283][0] , taskNeigh[284][0] , taskNeigh[285][0] , taskNeigh[286][0] , taskNeigh[287][0] , taskNeigh[288][0] , taskNeigh[289][0] , taskNeigh[290][0] , taskNeigh[291][0] , taskNeigh[292][0] , taskNeigh[293][0] , taskNeigh[294][0] , taskNeigh[295][0] , taskNeigh[296][0] , taskNeigh[297][0] , taskNeigh[298][0] , taskNeigh[299][0] , taskNeigh[300][0] , taskNeigh[301][0] , taskNeigh[302][0] , taskNeigh[303][0] , taskNeigh[304][0] , taskNeigh[305][0] , taskNeigh[306][0] , taskNeigh[307][0] , taskNeigh[308][0] , taskNeigh[309][0] , taskNeigh[310][0] , taskNeigh[311][0] , taskNeigh[312][0] , taskNeigh[313][0] , taskNeigh[314][0] , taskNeigh[315][0] , taskNeigh[316][0] , taskNeigh[317][0] , taskNeigh[318][0] , taskNeigh[319][0] , taskNeigh[320][0] , taskNeigh[321][0] , taskNeigh[322][0] , taskNeigh[323][0] , taskNeigh[324][0] , taskNeigh[325][0] , taskNeigh[326][0] , taskNeigh[327][0] , taskNeigh[328][0] , taskNeigh[329][0] , taskNeigh[330][0] , taskNeigh[331][0] , taskNeigh[332][0] , taskNeigh[333][0] , taskNeigh[334][0] , taskNeigh[335][0] , taskNeigh[336][0] , taskNeigh[337][0] , taskNeigh[338][0] , taskNeigh[339][0] , taskNeigh[340][0] , taskNeigh[341][0] ) priority_if_supported(idxLevel==FAbstractAlgorithm::lowerWorkingLevel-1?FFmmAlgorithmOmp4_Prio_M2L:FFmmAlgorithmOmp4_Prio_M2L_High)
                    {
                      kernels[omp_get_thread_num()]->M2L(  taskCell, taskNeigh, neighborPositions, counter, idxLevel);
                    }
                }
            } while(octreeIterator.moveRight());
            ////////////////////////////////////////////////////////////////
            // move up  and goto left
            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (M2L) = "  << counterTime.tacAndElapsed() << " s)\n" );

    }

    /////////////////////////////////////////////////////////////////////////////
    // Downward
    /////////////////////////////////////////////////////////////////////////////

    void downardPass(){ // second L2L
        FLOG( FLog::Controller.write("\tStart Downward Pass (L2L)\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.moveDown();

        for(int idxLevel = 2 ; idxLevel < FAbstractAlgorithm::upperWorkingLevel ; ++idxLevel){
            octreeIterator.moveDown();
        }

        typename OctreeClass::Iterator avoidGotoLeftIterator(octreeIterator);

        const int heightMinusOne = FAbstractAlgorithm::lowerWorkingLevel - 1;
        // for each levels exepted leaf level
        for(int idxLevel = FAbstractAlgorithm::upperWorkingLevel ; idxLevel < heightMinusOne ; ++idxLevel ){
            FLOG(FTic counterTimeLevel);
            // for each cells
            do{
                CellClass* taskCell = octreeIterator.getCurrentCell();
                CellClass* taskChild[8];
                memcpy(taskChild, octreeIterator.getCurrentChild(), 8*sizeof(CellClass*));

                const unsigned char* taskCellLocal = getCellLocalDepPtr(taskCell);
                const unsigned char* taskChildLocal[8] = {nullptr};
                int counterFill = 0;
                for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                    if(taskChild[idxChild]){
                        taskChildLocal[counterFill++] = getCellLocalDepPtr(taskChild[idxChild]);
                    }
                }


                switch(counterFill){
                case 1:
                    #pragma omp task firstprivate(taskCell, taskCellLocal, taskChild, taskChildLocal, idxLevel) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskChildLocal[0][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2L)
                    {
                    kernels[omp_get_thread_num()]->L2L( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 2:
                    #pragma omp task firstprivate(taskCell, taskCellLocal, taskChild, taskChildLocal, idxLevel) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskChildLocal[0][0],taskChildLocal[1][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2L)
                    {
                    kernels[omp_get_thread_num()]->L2L( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 3:
                    #pragma omp task firstprivate(taskCell, taskCellLocal, taskChild, taskChildLocal, idxLevel) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskChildLocal[0][0],taskChildLocal[1][0],taskChildLocal[2][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2L)
                    {
                    kernels[omp_get_thread_num()]->L2L( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 4:
                    #pragma omp task firstprivate(taskCell, taskCellLocal, taskChild, taskChildLocal, idxLevel) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskChildLocal[0][0],taskChildLocal[1][0],taskChildLocal[2][0],taskChildLocal[3][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2L)
                    {
                    kernels[omp_get_thread_num()]->L2L( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 5:
                    #pragma omp task firstprivate(taskCell, taskCellLocal, taskChild, taskChildLocal, idxLevel) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskChildLocal[0][0],taskChildLocal[1][0],taskChildLocal[2][0],taskChildLocal[3][0],taskChildLocal[4][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2L)
                    {
                    kernels[omp_get_thread_num()]->L2L( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 6:
                    #pragma omp task firstprivate(taskCell, taskCellLocal, taskChild, taskChildLocal, idxLevel) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskChildLocal[0][0],taskChildLocal[1][0],taskChildLocal[2][0],taskChildLocal[3][0],taskChildLocal[4][0],taskChildLocal[5][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2L)
                    {
                    kernels[omp_get_thread_num()]->L2L( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 7:
                    #pragma omp task firstprivate(taskCell, taskCellLocal, taskChild, taskChildLocal, idxLevel) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskChildLocal[0][0],taskChildLocal[1][0],taskChildLocal[2][0],taskChildLocal[3][0],taskChildLocal[4][0],taskChildLocal[5][0],taskChildLocal[6][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2L)
                    {
                    kernels[omp_get_thread_num()]->L2L( taskCell , taskChild, idxLevel);
                    }
                    break;
                case 8:
                    #pragma omp task firstprivate(taskCell, taskCellLocal, taskChild, taskChildLocal, idxLevel) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskChildLocal[0][0],taskChildLocal[1][0],taskChildLocal[2][0],taskChildLocal[3][0],taskChildLocal[4][0],taskChildLocal[5][0],taskChildLocal[6][0],taskChildLocal[7][0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2L)
                    {
                    kernels[omp_get_thread_num()]->L2L( taskCell , taskChild, idxLevel);
                    }
                    break;
                default:
                    FAssertLF(0, "Thus must not be possible");
                }
            } while(octreeIterator.moveRight());

            avoidGotoLeftIterator.moveDown();
            octreeIterator = avoidGotoLeftIterator;

            FLOG( FLog::Controller << "\t\t>> Level " << idxLevel << " = "  << counterTimeLevel.tacAndElapsed() << " s\n" );
        }

        FLOG( FLog::Controller << "\tFinished (@Downward Pass (L2L) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }


    /////////////////////////////////////////////////////////////////////////////
    // Direct
    /////////////////////////////////////////////////////////////////////////////

    /** P2P */
    void directPass(){
        FLOG( FLog::Controller.write("\tStart Direct Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);
        const int heightMinusOne = OctreeHeight - 1;

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();

        // for each leafs
        do{
            // There is a maximum of 26 neighbors
            ContainerClass* neighbors[26];
            int neighborPositions[26];
            const int counter = tree->getLeafsNeighbors(neighbors, neighborPositions, octreeIterator.getCurrentGlobalCoordinate(),heightMinusOne);

            ContainerClass* taskParticlesTgt = octreeIterator.getCurrentListTargets();
            ContainerClass* taskParticlesSrc = octreeIterator.getCurrentListSrc();
            const FTreeCoordinate coord = octreeIterator.getCurrentGlobalCoordinate();

            switch(counter){
            case 0:
            if(taskParticlesTgt == taskParticlesSrc){
                #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                {
                    kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                            taskParticlesTgt, neighbors, neighborPositions, counter);
                }
            }
            else{
                #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                {
                    kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                            taskParticlesSrc, neighbors, neighborPositions, counter);
                }
            }
                break;
            case 1:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 2:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 3:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 4:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 5:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 6:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 7:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 8:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 9:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 10:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 11:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 12:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 13:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 14:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 15:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 16:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 17:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 18:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 19:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 20:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 21:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 22:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 23:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0], neighbors[22][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0], neighbors[22][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 24:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0], neighbors[22][0], neighbors[23][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0], neighbors[22][0], neighbors[23][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 25:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0], neighbors[22][0], neighbors[23][0], neighbors[24][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0], neighbors[22][0], neighbors[23][0], neighbors[24][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            case 26:
                if(taskParticlesTgt == taskParticlesSrc){
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, coord) depend(commute_if_supported:taskParticlesTgt[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0], neighbors[22][0], neighbors[23][0], neighbors[24][0], neighbors[25][0] ) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesTgt, neighbors, neighborPositions, counter);
                    }
                }
                else{
                    #pragma omp task firstprivate(neighbors, neighborPositions, counter, taskParticlesTgt, taskParticlesSrc, coord) depend(commute_if_supported:taskParticlesTgt[0]) depend(in:taskParticlesSrc[0], neighbors[0][0], neighbors[1][0], neighbors[2][0], neighbors[3][0], neighbors[4][0], neighbors[5][0], neighbors[6][0], neighbors[7][0], neighbors[8][0], neighbors[9][0], neighbors[10][0], neighbors[11][0], neighbors[12][0], neighbors[13][0], neighbors[14][0], neighbors[15][0], neighbors[16][0], neighbors[17][0], neighbors[18][0], neighbors[19][0], neighbors[20][0], neighbors[21][0], neighbors[22][0], neighbors[23][0], neighbors[24][0], neighbors[25][0]) priority_if_supported((taskParticlesTgt->getNbParticles())>size_t(p2pPrioCriteria*1.1)?FFmmAlgorithmOmp4_Prio_P2P_Big:FFmmAlgorithmOmp4_Prio_P2P_Small)
                    {
                        kernels[omp_get_thread_num()]->P2P(coord, taskParticlesTgt,
                                taskParticlesSrc, neighbors, neighborPositions, counter);
                    }
                }
                 break;
            default:
                FAssertLF(0, "This must be impossible");
        }
        } while(octreeIterator.moveRight());

        FLOG( FLog::Controller << "\tFinished (@Direct Pass (P2P) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

    /** L2P */
    void mergePass(){
        FLOG( FLog::Controller.write("\tStart Direct Pass\n").write(FLog::Flush); );
        FLOG(FTic counterTime);

        typename OctreeClass::Iterator octreeIterator(tree);
        octreeIterator.gotoBottomLeft();

        // for each leafs
        do{
            CellClass* taskCell = octreeIterator.getCurrentCell();
            const unsigned char* taskCellLocal = getCellLocalDepPtr(taskCell);
            ContainerClass* taskParticles = octreeIterator.getCurrentListTargets();
            #pragma omp task firstprivate(taskCell,taskCellLocal, taskParticles) depend(in:taskCellLocal[0]) depend(commute_if_supported:taskParticles[0]) priority_if_supported(FFmmAlgorithmOmp4_Prio_L2P)
            {
                kernels[omp_get_thread_num()]->L2P(taskCell, taskParticles);
            }
        } while(octreeIterator.moveRight());


        FLOG( FLog::Controller << "\tFinished (@Merge Pass (L2P) = "  << counterTime.tacAndElapsed() << " s)\n" );
    }

};


#endif // FFMMALGORITHMOMP4_HPP
