// See LICENCE file at project root

/**
 * This file provide an interface to the Chebyshev kernel, in order to
 * call it from C code (and thus use it through API's user defined
 * kernel feature).
 */

#ifndef FCHEBINTERFACE_H
#define FCHEBINTERFACE_H


//To access leaf datas
struct FChebLeaf_struct;
typedef struct FChebLeaf_struct ChebLeafStruct;

//To access a cell
struct FChebCell_struct;
typedef struct FChebCell_struct ChebCellStruct;
ChebCellStruct * ChebCellStruct_create(long long int index,int * tree_position);
void ChebCellStruct_free(ChebCellStruct * cell);

//To manage leaves
ChebLeafStruct * ChebLeafStruct_create(FSize nbPart);
void ChebLeafStruct_free(void * leafData);
void ChebLeafStruct_fill(FSize nbPart, const FSize * idxPart,
                         long long morton_index, void * leafData,
                         void * userData);

void ChebLeafStruct_get_back_results(void * leafData,
                                     double ** forceXptr,  double ** forceYptr,  double ** forceZptr,
                                     double ** potentialsptr);

//To access the kernel
struct FChebKernel_struct;
typedef struct FChebKernel_struct ChebKernelStruct;
ChebKernelStruct * ChebKernelStruct_create(int inTreeHeight,
                                           double inBoxWidth,
                                           double* inBoxCenter);

void ChebKernelStruct_free(void * kernel);
//To access kernel member function

void ChebKernel_P2M(void * leafCell, void * leafData, FSize nbParticles,const FSize* particleIndexes, void* kernel);
void ChebKernel_M2M(int level, void* parentCell, int childPosition, void* childCell, void* kernel);
//Change here to somethong nearer M2L defined in Src/Components/FAbstractKernels.hpp
void ChebKernel_M2L(int level, void* targetCell,const int*neighborPositions,int size, void** sourceCell, void* kernel);
void ChebKernel_L2L(int level, void* parentCell, int childPosition, void* childCell, void* kernel);
void ChebKernel_L2P(void* leafCell, void * leafData, FSize nbParticles, const FSize* particleIndexes, void* kernel);
void ChebKernel_P2P(void * targetLeaf, FSize nbParticles, const FSize* particleIndexes,
                    void ** sourceLeaves,
                    const FSize ** sourceParticleIndexes, FSize * sourceNbPart,
                    const int * sourcePosition,const int size,void* userData);
void ChebKernel_P2P_old(void * targetLeaf, FSize nbParticles, const FSize* particleIndexes,
                        void ** sourceLeaves,
                        const FSize ** sourceParticleIndexes,FSize* sourceNbPart,
                        const int * sourcePosition,const int size, void* inKernel);

void ChebKernel_P2PRemote(void * targetLeaf,FSize nbParticles, const FSize* particleIndexes,
                          void ** sourceLeaves,
                          const FSize ** sourceParticleIndexes,FSize * sourceNbPart,
                          const int * sourcePosition, const int size, void* inKernel);

void ChebCell_reset(int level, long long morton_index, int* tree_position, double* spatial_position, void * userCell, void * kernel);

FSize ChebCell_getSize(int level, long long morton_index);

void ChebCell_copy(void * userDatas, FSize size, void * memoryAllocated);

void* ChebCell_restore(int level, void * arrayTobeRead);

FSize ChebLeaf_getSize(FSize nbPart);

void ChebLeaf_copy(FSize nbPart,void * userdata, void * memAllocated);

void * ChebLeaf_restore(FSize nbPart,void * memToRead);

typedef struct myUserDatas{
    ChebKernelStruct * kernelStruct;
    double * insertedPositions;
    double * myPhyValues;
    double ** forcesComputed;
    //In the same way we store multiples forces array.
    double ** potentials;
    double totalEnergy;
}UserData;


#endif //FCHEBINTERFACE_H
