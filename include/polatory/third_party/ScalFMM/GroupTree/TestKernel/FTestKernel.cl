/** This file contains the prototype for a kernel in opencl */
// @SCALFMM_PRIVATE


/***************************************************************************/
/***************************************************************************/
/************************CHANGE THINGS HERE*********************************/
/***************************************************************************/

typedef ___FSize___ FSize;
typedef ___FReal___ FReal;
typedef ___FParticleValueClass___ FParticleValueClass;
typedef long long int MortonIndex;

#define FOpenCLGroupOfCellsCellIsEmptyFlag  ((MortonIndex)-1)

#define NbAttributesPerParticle ___NbAttributesPerParticle___
#define NbSymbAttributes ___NbSymbAttributes___

#define FOpenCLGroupOfParticlesMemoryAlignementBytes  ___FP2PDefaultAlignement___
#define FOpenCLGroupOfParticlesMemoryAlignementParticles (FOpenCLGroupOfParticlesMemoryAlignementBytes/sizeof(FReal))
#define FOpenCLGroupOfParticlesLeafIsEmptyFlag ((MortonIndex)-1)

#define NULLPTR (0)

#define DefaultStructAlign ___DefaultStructAlign___

struct FSymboleCellClass {
    MortonIndex mortonIndex;
    int coordinates[3];
} __attribute__ ((aligned (DefaultStructAlign)));

typedef long long FTestCellPODData;
typedef FTestCellPODData FPoleCellClass;
typedef FTestCellPODData FLocalCellClass;

struct FWrappeCell{
    __global struct FSymboleCellClass* symb;
    __global FPoleCellClass* up;
    __global FLocalCellClass* down;
};


/***************************************************************************/
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

struct OutOfBlockInteraction{
    MortonIndex outIndex;
    MortonIndex insideIndex;
    int relativeOutPosition;
    int insideIdxInBlock;
    int outsideIdxInBlock;
} __attribute__ ((aligned (DefaultStructAlign)));

#define Between(inValue, inMin, inMax)  ( (inMin) <= (inValue) && (inValue) < (inMax) )
#define pow2(power)  (1 << (power))
#define Abs(inV) (inV < 0 ? -inV : inV)

int3 GetPositionFromMorton(MortonIndex inIndex, const int inLevel){
    MortonIndex mask = 0x1LL;

    int3 coord;
    coord.x = 0;
    coord.y = 0;
    coord.z = 0;

    for(int indexLevel = 0; indexLevel < inLevel ; ++indexLevel){
        coord.z |= (int)(inIndex & mask);
        inIndex >>= 1;
        coord.y |= (int)(inIndex & mask);
        inIndex >>= 1;
        coord.x |= (int)(inIndex & mask);

        mask <<= 1;
    }

    return coord;
}

MortonIndex GetMortonIndex(const int3 coord, const int inLevel) {
    MortonIndex index = 0x0LL;
    MortonIndex mask = 0x1LL;
    // the ordre is xyz.xyz...
    MortonIndex mx = coord.x << 2;
    MortonIndex my = coord.y << 1;
    MortonIndex mz = coord.z;

    for(int indexLevel = 0; indexLevel < inLevel ; ++indexLevel){
        index |= (mz & mask);
        mask <<= 1;
        index |= (my & mask);
        mask <<= 1;
        index |= (mx & mask);
        mask <<= 1;

        mz <<= 2;
        my <<= 2;
        mx <<= 2;
    }

    return index;
}

int GetNeighborsIndexes(const int3 coord, const int OctreeHeight, MortonIndex indexes[26], int indexInArray[26]) {
    int idxNeig = 0;
    int limite = 1 << (OctreeHeight - 1);
    // We test all cells around
    for(int idxX = -1 ; idxX <= 1 ; ++idxX){
        if(!Between(coord.x + idxX,0, limite)) continue;

        for(int idxY = -1 ; idxY <= 1 ; ++idxY){
            if(!Between(coord.y + idxY,0, limite)) continue;

            for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                if(!Between(coord.z + idxZ,0, limite)) continue;

                // if we are not on the current cell
                if( idxX || idxY || idxZ ){
                    int3 other;

                    other.x = coord.x + idxX;
                    other.y = coord.y + idxY;
                    other.z = coord.z + idxZ;

                    indexes[ idxNeig ] = GetMortonIndex(other, OctreeHeight - 1);
                    indexInArray[ idxNeig ] = ((idxX+1)*3 + (idxY+1)) * 3 + (idxZ+1);
                    ++idxNeig;
                }
            }
        }
    }
    return idxNeig;
}

int GetInteractionNeighbors(const int3 coord, const int inLevel, MortonIndex inNeighbors[189], int inNeighborsPosition[189]) {
    // Then take each child of the parent's neighbors if not in directNeighbors
    // Father coordinate
    int3 parentCell;
    parentCell.x = coord.x>>1;
    parentCell.y = coord.y>>1;
    parentCell.z = coord.z>>1;

    // Limite at parent level number of box (split by 2 by level)
    const int limite = pow2(inLevel-1);

    int idxNeighbors = 0;
    // We test all cells around
    for(int idxX = -1 ; idxX <= 1 ; ++idxX){
        if(!Between(parentCell.x + idxX,0,limite)) continue;

        for(int idxY = -1 ; idxY <= 1 ; ++idxY){
            if(!Between(parentCell.y + idxY,0,limite)) continue;

            for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                if(!Between(parentCell.z + idxZ,0,limite)) continue;

                // if we are not on the current cell
                if( idxX || idxY || idxZ ){
                    int3 otherParent;

                    otherParent.x = parentCell.x + idxX;
                    otherParent.y = parentCell.y + idxY;
                    otherParent.z = parentCell.z + idxZ;

                    const MortonIndex mortonOther = GetMortonIndex(otherParent, inLevel-1);

                    // For each child
                    for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                        const int xdiff  = ((otherParent.x<<1) | ( (idxCousin>>2) & 1)) - coord.x;
                        const int ydiff  = ((otherParent.y<<1) | ( (idxCousin>>1) & 1)) - coord.y;
                        const int zdiff  = ((otherParent.z<<1) | (idxCousin&1)) - coord.z;

                        // Test if it is a direct neighbor
                        if(Abs(xdiff) > 1 || Abs(ydiff) > 1 || Abs(zdiff) > 1){
                            // add to neighbors
                            inNeighborsPosition[idxNeighbors] = ((( (xdiff+3) * 7) + (ydiff+3))) * 7 + zdiff + 3;
                            inNeighbors[idxNeighbors++] = (mortonOther << 3) | idxCousin;
                        }
                    }
                }
            }
        }
    }

    return idxNeighbors;
}


void FSetToNullptr343(struct FWrappeCell ptrs[343]){
    int idx;
    for( idx = 0 ; idx < 343 ; ++idx){
        ptrs[idx].symb = NULLPTR;
    }
}

/***************************************************************************/
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/


struct FOpenCLGroupAttachedLeaf {
    //< Nb of particles in the current leaf
    FSize nbParticles;
    //< Pointers to the positions of the particles
    __global FReal* positionsPointers[3];
    //< Pointers to the attributes of the particles
    __global FParticleValueClass* attributes[NbSymbAttributes+NbAttributesPerParticle];
};

struct FOpenCLGroupAttachedLeaf BuildFOpenCLGroupAttachedLeaf(const FSize inNbParticles, __global FReal* inPositionBuffer, const size_t inLeadingPosition,
                                                              __global FParticleValueClass* inAttributesBuffer, const size_t inLeadingAttributes){
    struct FOpenCLGroupAttachedLeaf leaf;
    leaf.nbParticles = (inNbParticles);
    // Redirect pointers to position
    leaf.positionsPointers[0] = inPositionBuffer;
    leaf.positionsPointers[1] = (__global FReal*)(((__global unsigned char*)inPositionBuffer) + inLeadingPosition);
    leaf.positionsPointers[2] = (__global FReal*)(((__global unsigned char*)inPositionBuffer) + inLeadingPosition*2);

    for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes ; ++idxAttribute){
        leaf.attributes[idxAttribute] =  (__global FParticleValueClass*)(((__global unsigned char*)inPositionBuffer) + inLeadingPosition*(idxAttribute+3));
    }

    // Redirect pointers to data
    if(inAttributesBuffer){
        for(unsigned idxAttribute = 0 ; idxAttribute < NbAttributesPerParticle ; ++idxAttribute){
            leaf.attributes[idxAttribute+NbSymbAttributes] = (__global FParticleValueClass*)(((__global unsigned char*)inAttributesBuffer) + idxAttribute*inLeadingAttributes);
        }
    }
    else{
        for(unsigned idxAttribute = 0 ; idxAttribute < NbAttributesPerParticle ; ++idxAttribute){
            leaf.attributes[idxAttribute+NbSymbAttributes] = NULLPTR;
        }
    }
    return leaf;
}

struct FOpenCLGroupAttachedLeaf EmptyFOpenCLGroupAttachedLeaf(){
    struct FOpenCLGroupAttachedLeaf leaf;
    leaf.nbParticles = -1;
    // Redirect pointers to position
    leaf.positionsPointers[0] = NULLPTR;
    leaf.positionsPointers[1] = NULLPTR;
    leaf.positionsPointers[2] = NULLPTR;

    // Redirect pointers to data
    for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes+NbAttributesPerParticle ; ++idxAttribute){
        leaf.attributes[idxAttribute] = NULLPTR;
    }
    return leaf;
}

bool FOpenCLGroupAttachedLeaf_isAttachedToSomething(const struct FOpenCLGroupAttachedLeaf* group){
    return (group->nbParticles != -1);
}
bool FOpenCLGroupAttachedLeaf_getNbParticles(const struct FOpenCLGroupAttachedLeaf* group){
    return (group->nbParticles);
}


/** One header is allocated at the beginning of each block */
struct FOpenCLGroupOfParticlesBlockHeader{
    MortonIndex startingIndex;
    MortonIndex endingIndex;
    int numberOfLeavesInBlock;

    //< The real number of particles allocated
    FSize nbParticlesAllocatedInGroup;
    //< Starting point of position
    size_t offsetPosition;
    //< Bytes difference/offset between position
    size_t positionsLeadingDim;
    //< Bytes difference/offset between attributes
    size_t attributeLeadingDim;
    //< The total number of particles in the group
    FSize nbParticlesInGroup;
}__attribute__ ((aligned (DefaultStructAlign)));

/** Information about a leaf */
struct FOpenCLGroupOfParticlesLeafHeader {
    MortonIndex mindex;
    FSize nbParticles;
    size_t offSet;
}__attribute__ ((aligned (DefaultStructAlign)));


struct FOpenCLGroupOfParticles {
    //< The size of memoryBuffer in byte
    size_t allocatedMemoryInByte;
    //< Pointer to a block memory
    __global unsigned char* memoryBuffer;

    //< Pointer to the header inside the block memory
    __global struct FOpenCLGroupOfParticlesBlockHeader*    blockHeader;
    //< Pointer to leaves information
    __global struct FOpenCLGroupOfParticlesLeafHeader*     leafHeader;
    //< The total number of particles in the group
    const FSize nbParticlesInGroup;

    //< Pointers to particle position x, y, z
    __global FReal* particlePosition[3];

    //< Pointers to the particles data inside the block memory
    __global FParticleValueClass*      attributesBuffer;
    __global FParticleValueClass*      particleAttributes[NbSymbAttributes+NbAttributesPerParticle];
};

struct FOpenCLGroupOfParticles BuildFOpenCLGroupOfParticles(__global unsigned char* inBuffer, const size_t inAllocatedMemoryInByte,
                                                            __global unsigned char* inAttributeBuffer){
    struct FOpenCLGroupOfParticles group;
    group.allocatedMemoryInByte = (inAllocatedMemoryInByte);
    group.memoryBuffer = (inBuffer);

    // Move the pointers to the correct position
    group.blockHeader         = ((__global struct FOpenCLGroupOfParticlesBlockHeader*)inBuffer);
    inBuffer += sizeof(struct FOpenCLGroupOfParticlesBlockHeader);
    group.leafHeader          = ((__global struct FOpenCLGroupOfParticlesLeafHeader*)inBuffer);

    // Init particle pointers
    // Assert group.blockHeader->positionsLeadingDim == (sizeof(FReal) * group.blockHeader->nbParticlesAllocatedInGroup);
    group.particlePosition[0] = (__global FReal*) (group.memoryBuffer + group.blockHeader->offsetPosition);
    group.particlePosition[1] = (group.particlePosition[0] + group.blockHeader->nbParticlesAllocatedInGroup);
    group.particlePosition[2] = (group.particlePosition[1] + group.blockHeader->nbParticlesAllocatedInGroup);

    // Redirect pointer to data
    // Assert group.blockHeader->attributeLeadingDim == (sizeof(FParticleValueClass) * group.blockHeader->nbParticlesAllocatedInGroup);
    __global unsigned char* previousPointer = ((__global unsigned char*)(group.particlePosition[2] + group.blockHeader->nbParticlesAllocatedInGroup));
    for(unsigned idxAttribute = 0 ; idxAttribute < NbSymbAttributes ; ++idxAttribute){
        group.particleAttributes[idxAttribute] = ((__global FParticleValueClass*)previousPointer);
        previousPointer += sizeof(FParticleValueClass)*group.blockHeader->nbParticlesAllocatedInGroup;
    }
    
    if(inAttributeBuffer){
        group.attributesBuffer = (__global FParticleValueClass*)inAttributeBuffer;
        for(unsigned idxAttribute = 0 ; idxAttribute < NbAttributesPerParticle ; ++idxAttribute){
            group.particleAttributes[idxAttribute+NbSymbAttributes] = ((__global FParticleValueClass*)inAttributeBuffer);
            inAttributeBuffer += sizeof(FParticleValueClass)*group.blockHeader->nbParticlesAllocatedInGroup;
        }
    }
    else{
        group.attributesBuffer = NULLPTR;
        for(unsigned idxAttribute = 0 ; idxAttribute < NbAttributesPerParticle ; ++idxAttribute){
            group.particleAttributes[idxAttribute+NbSymbAttributes] = NULLPTR;
        }
    }
    
    return group;
}
MortonIndex FOpenCLGroupOfParticles_getStartingIndex(const struct FOpenCLGroupOfParticles* group) {
    return group->blockHeader->startingIndex;
}
MortonIndex FOpenCLGroupOfParticles_getEndingIndex(const struct FOpenCLGroupOfParticles* group) {
    return group->blockHeader->endingIndex;
}
int FOpenCLGroupOfParticles_getNumberOfLeaves(const struct FOpenCLGroupOfParticles* group) {
    return group->blockHeader->numberOfLeavesInBlock;
}
bool FOpenCLGroupOfParticles_isInside(const struct FOpenCLGroupOfParticles* group, const MortonIndex inIndex) {
    return group->blockHeader->startingIndex <= inIndex && inIndex < group->blockHeader->endingIndex;
}


/** Return the idx in array of the cell */
MortonIndex FOpenCLGroupOfParticles_getLeafMortonIndex(const struct FOpenCLGroupOfParticles* group, const int id){
    return group->leafHeader[id].mindex;
}

/** Check if a cell exist (by binary search) and return it index */
int FOpenCLGroupOfParticles_getLeafIndex(const struct FOpenCLGroupOfParticles* group, const MortonIndex leafIdx){
    int idxLeft = 0;
    int idxRight = group->blockHeader->numberOfLeavesInBlock-1;
    while(idxLeft <= idxRight){
        const int idxMiddle = (idxLeft+idxRight)/2;
        if(group->leafHeader[idxMiddle].mindex == leafIdx){
            return idxMiddle;
        }
        if(leafIdx < group->leafHeader[idxMiddle].mindex){
            idxRight = idxMiddle-1;
        }
        else{
            idxLeft = idxMiddle+1;
        }
    }
    return -1;
}


bool FOpenCLGroupOfParticles_exists(const struct FOpenCLGroupOfParticles* group, const MortonIndex inIndex) {
    return FOpenCLGroupOfParticles_isInside(group, inIndex) && (FOpenCLGroupOfParticles_getLeafIndex(group, inIndex) != -1);
}
struct FOpenCLGroupAttachedLeaf FOpenCLGroupOfParticles_getLeaf(struct FOpenCLGroupOfParticles* group, const int id){
    return BuildFOpenCLGroupAttachedLeaf(group->leafHeader[id].nbParticles,
                                         group->particlePosition[0] + group->leafHeader[id].offSet,
            group->blockHeader->positionsLeadingDim,
            (group->attributesBuffer?group->particleAttributes[NbSymbAttributes] + group->leafHeader[id].offSet:NULLPTR),
            group->blockHeader->attributeLeadingDim);
}


struct FOpenCLGroupOfCellsBlockHeader{
    MortonIndex startingIndex;
    MortonIndex endingIndex;
    int numberOfCellsInBlock;
} __attribute__ ((aligned (DefaultStructAlign)));


struct FOpenCLGroupOfCells {
    //< The size of the memoryBuffer
    size_t allocatedMemoryInByte;
    //< Pointer to a block memory
    __global unsigned char* memoryBuffer;

    //< Pointer to the header inside the block memory
    __global struct FOpenCLGroupOfCellsBlockHeader*    blockHeader;
    //< Pointer to the indexes table inside the block memory
    __global MortonIndex*    cellIndexes;
    //< Pointer to the cells inside the block memory
    __global struct FSymboleCellClass*      blockCells;
    
    //< The multipole data
    __global FPoleCellClass* cellMultipoles;
    //< The local data
    __global FLocalCellClass* cellLocals;
};

struct FOpenCLGroupOfCells BuildFOpenCLGroupOfCells(__global unsigned char* inBuffer, const size_t inAllocatedMemoryInByte,
                                                    __global unsigned char* inCellMultipoles, __global unsigned char* inCellLocals){
    struct FOpenCLGroupOfCells group;
    group.memoryBuffer = (inBuffer);
    group.allocatedMemoryInByte = (inAllocatedMemoryInByte);

    // Move the pointers to the correct position
    group.blockHeader         = (__global struct FOpenCLGroupOfCellsBlockHeader*)(inBuffer);
    inBuffer += sizeof(struct FOpenCLGroupOfCellsBlockHeader);
    group.cellIndexes   = (__global MortonIndex*)(inBuffer);
    inBuffer += (group.blockHeader->numberOfCellsInBlock*sizeof(MortonIndex));
    group.blockCells          = (__global struct FSymboleCellClass*)(inBuffer);
    inBuffer += (group.blockHeader->numberOfCellsInBlock*sizeof(struct FSymboleCellClass));
    // Assert(((size_t)(inBuffer-group.memoryBuffer) == inAllocatedMemoryInByte);

    group.cellMultipoles = (__global FPoleCellClass*)inCellMultipoles;
    group.cellLocals = (__global FLocalCellClass*)inCellLocals;
    return group;
}
MortonIndex FOpenCLGroupOfCells_getStartingIndex(const struct FOpenCLGroupOfCells* group) {
    return group->blockHeader->startingIndex;
}
MortonIndex FOpenCLGroupOfCells_getEndingIndex(const struct FOpenCLGroupOfCells* group) {
    return group->blockHeader->endingIndex;
}
int FOpenCLGroupOfCells_getNumberOfCellsInBlock(const struct FOpenCLGroupOfCells* group) {
    return group->blockHeader->numberOfCellsInBlock;
}
MortonIndex FOpenCLGroupOfCells_getSizeOfInterval(const struct FOpenCLGroupOfCells* group) {
    return group->blockHeader->endingIndex - group->blockHeader->startingIndex;
}
bool FOpenCLGroupOfCells_isInside(const struct FOpenCLGroupOfCells* group, const MortonIndex inIndex){
    return group->blockHeader->startingIndex <= inIndex && inIndex < group->blockHeader->endingIndex;
}

MortonIndex FOpenCLGroupOfCells_getCellMortonIndex(const struct FOpenCLGroupOfCells* group,const int cellPos){
    return group->cellIndexes[cellPos];
}

int FOpenCLGroupOfCells_getCellIndex(const struct FOpenCLGroupOfCells* group,const MortonIndex cellIdx){
    int idxLeft = 0;
    int idxRight = group->blockHeader->numberOfCellsInBlock-1;
    while(idxLeft <= idxRight){
        const int idxMiddle = (idxLeft+idxRight)/2;
        if(group->cellIndexes[idxMiddle] == cellIdx){
            return idxMiddle;
        }
        if(cellIdx < group->cellIndexes[idxMiddle]){
            idxRight = idxMiddle-1;
        }
        else{
            idxLeft = idxMiddle+1;
        }
    }
    return -1;
}

int FOpenCLGroupOfCells_getFistChildIdx(const struct FOpenCLGroupOfCells* group, const MortonIndex parentIdx) {
    int idxLeft = 0;
    int idxRight = group->blockHeader->numberOfCellsInBlock-1;
    while(idxLeft <= idxRight){
        int idxMiddle = (idxLeft+idxRight)/2;
        if((group->cellIndexes[idxMiddle]>>3) == parentIdx){
            while(0 < idxMiddle && (group->cellIndexes[idxMiddle-1]>>3) == parentIdx){
                idxMiddle -= 1;
            }
            return idxMiddle;
        }
        if(parentIdx < (group->cellIndexes[idxMiddle]>>3)){
            idxRight = idxMiddle-1;
        }
        else{
            idxLeft = idxMiddle+1;
        }
    }
    return -1;
}


bool FOpenCLGroupOfCells_exists(const struct FOpenCLGroupOfCells* group, const MortonIndex inIndex) {
    return FOpenCLGroupOfCells_isInside(group, inIndex) && FOpenCLGroupOfCells_getCellIndex(group, inIndex) != -1;
}
struct FWrappeCell FOpenCLGroupOfCells_getCompleteCell(struct FOpenCLGroupOfCells* group, const int cellPos){
    struct FWrappeCell cell;
    cell.symb = &group->blockCells[cellPos];
    cell.up = &group->cellMultipoles[cellPos];
    cell.down = &group->cellLocals[cellPos];
    return cell;
}

struct FWrappeCell FOpenCLGroupOfCells_getUpCell(struct FOpenCLGroupOfCells* group, const int cellPos){
    struct FWrappeCell cell;
    cell.symb = &group->blockCells[cellPos];
    cell.up = &group->cellMultipoles[cellPos];
    cell.down = NULLPTR;
    return cell;
}

struct FWrappeCell FOpenCLGroupOfCells_getDownCell(struct FOpenCLGroupOfCells* group, const int cellPos){
    struct FWrappeCell cell;
    cell.symb = &group->blockCells[cellPos];
    cell.up = NULLPTR;
    cell.down =&group->cellLocals[cellPos];
    return cell;
}

struct Uptr9{
    __global unsigned char* ptrs[9];
} __attribute__ ((aligned (DefaultStructAlign)));

struct size_t9{
    size_t v[9];
} __attribute__ ((aligned (DefaultStructAlign)));

struct Uptr343{
    __global unsigned char* ptrs[343];
};

/***************************************************************************/
/***************************************************************************/
/************************CHANGE THINGS HERE*********************************/
/***************************************************************************/


void P2M(struct FWrappeCell pole, const struct FOpenCLGroupAttachedLeaf particles, __global void* user_data) {
    *pole.up = particles.nbParticles;
}

void M2M(struct FWrappeCell  pole, struct FWrappeCell child[8], const int level, __global void* user_data) {
    for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
        if(child[idxChild].symb){
            *pole.up += *child[idxChild].up;
        }
    }
}

void M2L(struct FWrappeCell const pole, const struct FWrappeCell* distantNeighbors,
         const int* relativePositions, const int size, const int level, __global void* user_data) {
    for(int idxNeigh = 0 ; idxNeigh < size ; ++idxNeigh){
        *pole.down += *distantNeighbors[idxNeigh].up;
    }
}

void L2L(const struct FWrappeCell localCell, struct FWrappeCell child[8], const int level, __global void* user_data) {
    for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
        if(child[idxChild].symb){
            *child[idxChild].down += *localCell.down;
        }
    }
}

void L2P(const struct FWrappeCell localCell, struct FOpenCLGroupAttachedLeaf particles, __global void* user_data){
    __global long long* partdown = particles.attributes[0];
    for(FSize idxPart = 0 ; idxPart < particles.nbParticles ; ++idxPart){
        partdown[idxPart] += *localCell.down;
    }
}

void P2P(const int3 pos,
         struct FOpenCLGroupAttachedLeaf  targets, const struct FOpenCLGroupAttachedLeaf sources,
         struct FOpenCLGroupAttachedLeaf directNeighborsParticles[27], int directNeighborsPositions[27], const int counter, __global void* user_data){
    long long cumul = sources.nbParticles-1;
    
    for(int idxNeigh = 0 ; idxNeigh < counter ; ++idxNeigh){
        if(FOpenCLGroupAttachedLeaf_isAttachedToSomething(&directNeighborsParticles[idxNeigh])){
            cumul += directNeighborsParticles[idxNeigh].nbParticles;
        }
    }
    
    __global long long* partdown = targets.attributes[0];
    for(FSize idxPart = 0 ; idxPart < targets.nbParticles ; ++idxPart){
        partdown[idxPart] += cumul;
    }
}

void P2PRemote(const int3 pos,
               struct FOpenCLGroupAttachedLeaf  targets, const struct FOpenCLGroupAttachedLeaf  sources,
               struct FOpenCLGroupAttachedLeaf directNeighborsParticles, const int position, __global void* user_data){
    __global long long* partdown = targets.attributes[0];
    for(FSize idxPart = 0 ; idxPart < targets.nbParticles ; ++idxPart){
        partdown[idxPart] += directNeighborsParticles.nbParticles;
    }
}

void P2POuter(const int3 pos,
               struct FOpenCLGroupAttachedLeaf  targets, const struct FOpenCLGroupAttachedLeaf  sources,
               struct FOpenCLGroupAttachedLeaf directNeighborsParticles, const int position, __global void* user_data){
    __global long long* partdown = targets.attributes[0];
    for(FSize idxPart = 0 ; idxPart < targets.nbParticles ; ++idxPart){
        partdown[idxPart] += directNeighborsParticles.nbParticles;
    }
}

int3 getCoordinate(const struct FWrappeCell cell) {
    int3 coord;
    coord.x = cell.symb->coordinates[0];
    coord.y = cell.symb->coordinates[1];
    coord.z = cell.symb->coordinates[2];
    return coord;
}


/***************************************************************************/
/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

#define FOpenCLCheck( test ) { FOpenCLCheckCore((test), __FILE__, __LINE__); }
#define FOpenCLCheckAfterCall() { FOpenCLCheckCore((cudaGetLastError()), __FILE__, __LINE__); }
#define FOpenCLAssertLF(ARGS) if(!(ARGS)){ *((char*)0x09) = 'e'; }
//#define FOpenCLAssertLF(ARGS) ARGS;

#define FMGetOppositeNeighIndex(index) (27-(index)-1)
#define FMGetOppositeInterIndex(index) (343-(index)-1)

#define FOpenCLMax(x,y) ((x)<(y) ? (y) : (x))
#define FOpenCLMin(x,y) ((x)>(y) ? (y) : (x))


__kernel void FOpenCL__bottomPassPerform(__global unsigned char* leafCellsPtr, size_t leafCellsSize,__global unsigned char* leafCellsUpPtr,
                                         __global unsigned char* containersPtr, size_t containersSize,
                                         __global void* userkernel ){
    struct FOpenCLGroupOfCells leafCells = BuildFOpenCLGroupOfCells(leafCellsPtr, leafCellsSize, leafCellsUpPtr, NULLPTR);
    struct FOpenCLGroupOfParticles containers = BuildFOpenCLGroupOfParticles(containersPtr, containersSize, NULLPTR);

    const int nbLeaves = FOpenCLGroupOfCells_getNumberOfCellsInBlock(&leafCells);
    
    for(int idxLeaf = 0 ; idxLeaf < nbLeaves ; ++idxLeaf){
        struct FWrappeCell cell = FOpenCLGroupOfCells_getUpCell(&leafCells, idxLeaf);
        FOpenCLAssertLF(cell.symb->mortonIndex == FOpenCLGroupOfCells_getCellMortonIndex(&leafCells, idxLeaf));
        struct FOpenCLGroupAttachedLeaf particles = FOpenCLGroupOfParticles_getLeaf(&containers, idxLeaf);
        FOpenCLAssertLF(FOpenCLGroupOfParticles_getLeafMortonIndex(&containers, idxLeaf) == FOpenCLGroupOfCells_getCellMortonIndex(&leafCells, idxLeaf));
        P2M(cell, particles, userkernel);
    }
}


/////////////////////////////////////////////////////////////////////////////////////
/// Upward Pass
/////////////////////////////////////////////////////////////////////////////////////

__kernel void FOpenCL__upwardPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsUpPtr,
                                         __global unsigned char* childCellsPtr, size_t childCellsSize, __global unsigned char* childCellsUpPtr,
                                         int idxLevel, __global void* userkernel){
    struct FOpenCLGroupOfCells currentCells = BuildFOpenCLGroupOfCells(currentCellsPtr, currentCellsSize, currentCellsUpPtr, NULLPTR);
    const int nbCells = FOpenCLGroupOfCells_getNumberOfCellsInBlock(&currentCells);
    struct FOpenCLGroupOfCells childCells = BuildFOpenCLGroupOfCells(childCellsPtr, childCellsSize, childCellsUpPtr, NULLPTR);
    const int childNbCells = FOpenCLGroupOfCells_getNumberOfCellsInBlock(&childCells);

    const MortonIndex firstParent = FOpenCLMax(FOpenCLGroupOfCells_getStartingIndex(&currentCells), FOpenCLGroupOfCells_getStartingIndex(&childCells)>>3);
    const MortonIndex lastParent = FOpenCLMin(FOpenCLGroupOfCells_getEndingIndex(&currentCells)-1, (FOpenCLGroupOfCells_getEndingIndex(&childCells)-1)>>3);

    int idxParentCell = FOpenCLGroupOfCells_getCellIndex(&currentCells,firstParent);
    int idxChildCell = FOpenCLGroupOfCells_getFistChildIdx(&childCells,firstParent);

    while(true){
        struct FWrappeCell cell = FOpenCLGroupOfCells_getUpCell(&currentCells, idxParentCell);
        struct FWrappeCell child[8];
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            child[idxChild].symb = NULLPTR;
        }

        do{
            const int idxChild = ((FOpenCLGroupOfCells_getCellMortonIndex(&childCells,idxChildCell)) & 7);
            child[idxChild] = FOpenCLGroupOfCells_getUpCell(&childCells, idxChildCell);
            idxChildCell += 1;
        }while(idxChildCell != childNbCells && cell.symb->mortonIndex == (FOpenCLGroupOfCells_getCellMortonIndex(&childCells, idxChildCell)>>3));

        M2M(cell, child, idxLevel, userkernel);

        if(FOpenCLGroupOfCells_getCellMortonIndex(&currentCells, idxParentCell) == lastParent){
            break;
        }

        idxParentCell += 1;
    }
}


/////////////////////////////////////////////////////////////////////////////////////
/// Transfer Pass Mpi
/////////////////////////////////////////////////////////////////////////////////////


__kernel  void FOpenCL__transferInoutPassPerformMpi(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsDownPtr,
                                                    __global unsigned char* externalCellsPtr, size_t externalCellsSize, __global unsigned char* externalCellsUpPtr,
                                                    int idxLevel, const __global struct OutOfBlockInteraction* outsideInteractions,
                                                    size_t nbOutsideInteractions, __global void* userkernel){
    struct FOpenCLGroupOfCells currentCells = BuildFOpenCLGroupOfCells(currentCellsPtr, currentCellsSize, NULLPTR, currentCellsDownPtr);
    struct FOpenCLGroupOfCells cellsOther = BuildFOpenCLGroupOfCells(externalCellsPtr, externalCellsSize, externalCellsUpPtr, NULLPTR);

    for(int outInterIdx = 0 ; outInterIdx < nbOutsideInteractions ; ++outInterIdx){
        const int cellPos = FOpenCLGroupOfCells_getCellIndex(&cellsOther, outsideInteractions[outInterIdx].outIndex);
        if(cellPos != -1){
            FOpenCLAssertLF(outsideInteractions[outInterIdx].outIndex == FOpenCLGroupOfCells_getCellMortonIndex(&cellsOther, outsideInteractions[outInterIdx].outIndex));
            struct FWrappeCell interCell = FOpenCLGroupOfCells_getUpCell(&cellsOther, cellPos);
            FOpenCLAssertLF(interCell.symb->mortonIndex == outsideInteractions[outInterIdx].outIndex);
            struct FWrappeCell cell = FOpenCLGroupOfCells_getDownCell(&currentCells, outsideInteractions[outInterIdx].insideIdxInBlock);
            FOpenCLAssertLF(cell.symb->mortonIndex == outsideInteractions[outInterIdx].insideIndex);

            const int relativeOutPosition = outsideInteractions[outInterIdx].relativeOutPosition;
            M2L( cell , &interCell, &relativeOutPosition,
                 1, idxLevel, userkernel);
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////
/// Transfer Pass
/////////////////////////////////////////////////////////////////////////////////////



__kernel  void FOpenCL__transferInPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize,
                                              __global unsigned char* currentCellsUpPtr, __global unsigned char* currentCellsDownPtr,
                                              int idxLevel, __global void* userkernel){
    struct FOpenCLGroupOfCells currentCells = BuildFOpenCLGroupOfCells(currentCellsPtr, currentCellsSize, currentCellsUpPtr, currentCellsDownPtr);

    const MortonIndex blockStartIdx = FOpenCLGroupOfCells_getStartingIndex(&currentCells);
    const MortonIndex blockEndIdx = FOpenCLGroupOfCells_getEndingIndex(&currentCells);

    const int nbCells = FOpenCLGroupOfCells_getNumberOfCellsInBlock(&currentCells);

    for(int idxCell = 0 ; idxCell < nbCells ; ++idxCell){
        struct FWrappeCell cell = FOpenCLGroupOfCells_getDownCell(&currentCells, idxCell);
        FOpenCLAssertLF(cell.symb->mortonIndex == FOpenCLGroupOfCells_getCellMortonIndex(&currentCells, idxCell));
        MortonIndex interactionsIndexes[189];
        int interactionsPosition[189];
        const int3 coord = (getCoordinate(cell));
        int counter = GetInteractionNeighbors(coord, idxLevel,interactionsIndexes,interactionsPosition);

        struct FWrappeCell interactions[343];
        FSetToNullptr343(interactions);
        int counterExistingCell = 0;

        for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
            if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                const int cellPos = FOpenCLGroupOfCells_getCellIndex(&currentCells, interactionsIndexes[idxInter]);
                if(cellPos != -1){
                    struct FWrappeCell interCell = FOpenCLGroupOfCells_getUpCell(&currentCells, cellPos);
                    interactions[counterExistingCell] = interCell;
                    interactionsPosition[counterExistingCell] = interactionsPosition[idxInter];
                    counterExistingCell += 1;
                }
            }
        }

        M2L( cell , interactions, interactionsPosition,
             counterExistingCell, idxLevel, userkernel);
    }
}



__kernel void FOpenCL__transferInoutPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize,
                                                __global unsigned char*  currentCellsUpPtr,
                                                __global unsigned char* externalCellsPtr, size_t externalCellsSize,
                                                __global unsigned char* externalCellsDownPtr,
                                                int idxLevel, int mode, const __global struct OutOfBlockInteraction* outsideInteractions,
                                                size_t nbOutsideInteractions, __global void* userkernel){
    struct FOpenCLGroupOfCells currentCells = BuildFOpenCLGroupOfCells(currentCellsPtr, currentCellsSize, currentCellsUpPtr, NULLPTR);
    struct FOpenCLGroupOfCells cellsOther = BuildFOpenCLGroupOfCells(externalCellsPtr, externalCellsSize, NULLPTR, externalCellsDownPtr);

    if(mode == 1){
        for(int outInterIdx = 0 ; outInterIdx < nbOutsideInteractions ; ++outInterIdx){
            struct FWrappeCell interCell = FOpenCLGroupOfCells_getUpCell(&cellsOther, outsideInteractions[outInterIdx].outsideIdxInBlock);
            FOpenCLAssertLF(interCell.symb->mortonIndex == outsideInteractions[outInterIdx].outIndex);
            struct FWrappeCell cell = FOpenCLGroupOfCells_getDownCell(&currentCells, outsideInteractions[outInterIdx].insideIdxInBlock);
            FOpenCLAssertLF(cell.symb->mortonIndex == outsideInteractions[outInterIdx].insideIndex);

            const int relativeOutPosition = outsideInteractions[outInterIdx].relativeOutPosition;
            M2L( cell , &interCell, &relativeOutPosition,
                 1, idxLevel, userkernel);
        }
    }
    else{
        for(int outInterIdx = 0 ; outInterIdx < nbOutsideInteractions ; ++outInterIdx){
            struct FWrappeCell interCell = FOpenCLGroupOfCells_getDownCell(&cellsOther, outsideInteractions[outInterIdx].outsideIdxInBlock);
            FOpenCLAssertLF(interCell.symb->mortonIndex == outsideInteractions[outInterIdx].outIndex);
            struct FWrappeCell cell = FOpenCLGroupOfCells_getUpCell(&currentCells, outsideInteractions[outInterIdx].insideIdxInBlock);
            FOpenCLAssertLF(cell.symb->mortonIndex == outsideInteractions[outInterIdx].insideIndex);

            const int relativepos = FMGetOppositeInterIndex(outsideInteractions[outInterIdx].relativeOutPosition);
            M2L( interCell , &cell, &relativepos, 1, idxLevel, userkernel);
        }
    }
}



/////////////////////////////////////////////////////////////////////////////////////
/// Downard Pass
/////////////////////////////////////////////////////////////////////////////////////


__kernel void FOpenCL__downardPassPerform(__global unsigned char* currentCellsPtr, size_t currentCellsSize, __global unsigned char* currentCellsDownPtr,
                                          __global unsigned char* childCellsPtr, size_t childCellsSize, __global unsigned char* childCellsDownPtr,
                                          int idxLevel, __global void* userkernel){
    struct FOpenCLGroupOfCells currentCells = BuildFOpenCLGroupOfCells(currentCellsPtr, currentCellsSize, NULLPTR, currentCellsDownPtr);
    const int nbCells = FOpenCLGroupOfCells_getNumberOfCellsInBlock(&currentCells);
    struct FOpenCLGroupOfCells childCells = BuildFOpenCLGroupOfCells(childCellsPtr, childCellsSize, NULLPTR, childCellsDownPtr);
    const int childNbCells = FOpenCLGroupOfCells_getNumberOfCellsInBlock(&childCells);

    const MortonIndex firstParent = FOpenCLMax(FOpenCLGroupOfCells_getStartingIndex(&currentCells), FOpenCLGroupOfCells_getStartingIndex(&childCells)>>3);
    const MortonIndex lastParent = FOpenCLMin(FOpenCLGroupOfCells_getEndingIndex(&currentCells)-1, (FOpenCLGroupOfCells_getEndingIndex(&childCells)-1)>>3);

    int idxParentCell = FOpenCLGroupOfCells_getCellIndex(&currentCells,firstParent);
    int idxChildCell = FOpenCLGroupOfCells_getFistChildIdx(&childCells,firstParent);

    while(true){
        struct FWrappeCell cell = FOpenCLGroupOfCells_getDownCell(&currentCells, idxParentCell);
        struct FWrappeCell child[8];
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            child[idxChild].symb = NULLPTR;
        }

        do{
            const int idxChild = ((FOpenCLGroupOfCells_getCellMortonIndex(&childCells,idxChildCell)) & 7);
            child[idxChild] = FOpenCLGroupOfCells_getDownCell(&childCells, idxChildCell);
            idxChildCell += 1;
        }while(idxChildCell != childNbCells && cell.symb->mortonIndex == (FOpenCLGroupOfCells_getCellMortonIndex(&childCells, idxChildCell)>>3));

        L2L(cell, child, idxLevel, userkernel);

        if(FOpenCLGroupOfCells_getCellMortonIndex(&currentCells, idxParentCell) == lastParent){
            break;
        }

        idxParentCell += 1;
    }
}



/////////////////////////////////////////////////////////////////////////////////////
/// Direct Pass MPI
/////////////////////////////////////////////////////////////////////////////////////


__kernel void FOpenCL__directInoutPassPerformMpi(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,
                                                 __global unsigned char* externalContainersPtr, size_t externalContainersSize, __global unsigned char* outsideInteractionsCl,
                                                 const __global struct OutOfBlockInteraction* outsideInteractions,
                                                 size_t nbOutsideInteractions, const int treeHeight, __global void* userkernel){
    struct FOpenCLGroupOfParticles containers = BuildFOpenCLGroupOfParticles(containersPtr, containersSize, containersDownPtr);
    struct FOpenCLGroupOfParticles containersOther = BuildFOpenCLGroupOfParticles(externalContainersPtr, externalContainersSize, NULLPTR);

    for(int outInterIdx = 0 ; outInterIdx < nbOutsideInteractions ; ++outInterIdx){
        const int leafPos = FOpenCLGroupOfParticles_getLeafIndex(&containersOther, outsideInteractions[outInterIdx].outIndex);
        if(leafPos != -1){
            FOpenCLAssertLF(FOpenCLGroupOfParticles_getLeafMortonIndex(&containersOther, leafPos) == outsideInteractions[outInterIdx].outIndex);
            struct FOpenCLGroupAttachedLeaf interParticles = FOpenCLGroupOfParticles_getLeaf(&containersOther, leafPos);
            struct FOpenCLGroupAttachedLeaf particles = FOpenCLGroupOfParticles_getLeaf(&containers, outsideInteractions[outInterIdx].insideIdxInBlock);
            FOpenCLAssertLF(FOpenCLGroupOfParticles_getLeafMortonIndex(&containers, outsideInteractions[outInterIdx].insideIdxInBlock) == outsideInteractions[outInterIdx].insideIndex);

            P2PRemote( GetPositionFromMorton(outsideInteractions[outInterIdx].insideIndex, treeHeight-1), particles, particles ,
                       interParticles, outsideInteractions[outInterIdx].relativeOutPosition, userkernel);
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////
/// Direct Pass
/////////////////////////////////////////////////////////////////////////////////////



__kernel void FOpenCL__directInPassPerform(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,
                                           const int treeHeight, __global void* userkernel){
    struct FOpenCLGroupOfParticles containers = BuildFOpenCLGroupOfParticles(containersPtr, containersSize, containersDownPtr);

    const MortonIndex blockStartIdx = FOpenCLGroupOfParticles_getStartingIndex(&containers);
    const MortonIndex blockEndIdx = FOpenCLGroupOfParticles_getEndingIndex(&containers);

    const int nbLeaves = FOpenCLGroupOfParticles_getNumberOfLeaves(&containers);

    for(int idxLeaf = 0 ; idxLeaf < nbLeaves ; ++idxLeaf){
        struct FOpenCLGroupAttachedLeaf particles = FOpenCLGroupOfParticles_getLeaf(&containers, idxLeaf);
        MortonIndex interactionsIndexes[26];
        int interactionsPosition[26];
        const int3 coord = GetPositionFromMorton(FOpenCLGroupOfParticles_getLeafMortonIndex(&containers, idxLeaf), treeHeight-1);
        int counter = GetNeighborsIndexes(coord, treeHeight,interactionsIndexes,interactionsPosition);

        struct FOpenCLGroupAttachedLeaf interactionsObjects[27];
        int neighPosition[26];
        int counterExistingCell = 0;

        for(int idxInter = 0 ; idxInter < counter ; ++idxInter){
            if( blockStartIdx <= interactionsIndexes[idxInter] && interactionsIndexes[idxInter] < blockEndIdx ){
                const int leafPos = FOpenCLGroupOfParticles_getLeafIndex(&containers, interactionsIndexes[idxInter]);
                if(leafPos != -1){
                    FOpenCLAssertLF(FOpenCLGroupOfParticles_getLeafMortonIndex(&containers, leafPos) == interactionsIndexes[idxInter]);
                    interactionsObjects[counterExistingCell] = FOpenCLGroupOfParticles_getLeaf(&containers, leafPos);
                    neighPosition[counterExistingCell] = interactionsPosition[idxInter];
                    counterExistingCell += 1;
                }
            }
        }

        P2P( coord, particles, particles , interactionsObjects, neighPosition, counterExistingCell, userkernel);
    }
}



__kernel void FOpenCL__directInoutPassPerform(__global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,
                                              __global unsigned char* externalContainersPtr, size_t externalContainersSize, __global unsigned char* externalContainersDownPtr,
                                              const __global struct OutOfBlockInteraction* outsideInteractions,
                                              size_t nbOutsideInteractions, const int treeHeight, __global void* userkernel){
    struct FOpenCLGroupOfParticles containers = BuildFOpenCLGroupOfParticles(containersPtr, containersSize, containersDownPtr);
    struct FOpenCLGroupOfParticles containersOther = BuildFOpenCLGroupOfParticles(externalContainersPtr, externalContainersSize, externalContainersDownPtr);

    for(int outInterIdx = 0 ; outInterIdx < nbOutsideInteractions ; ++outInterIdx){
        const int leafPos = FOpenCLGroupOfParticles_getLeafIndex(&containersOther, outsideInteractions[outInterIdx].outIndex);
        if(leafPos != -1){
            struct FOpenCLGroupAttachedLeaf interParticles = FOpenCLGroupOfParticles_getLeaf(&containersOther, outsideInteractions[outInterIdx].outsideIdxInBlock);
            struct FOpenCLGroupAttachedLeaf particles = FOpenCLGroupOfParticles_getLeaf(&containers, outsideInteractions[outInterIdx].insideIdxInBlock);

            FOpenCLAssertLF(FOpenCLGroupOfParticles_getLeafMortonIndex(&containers, outsideInteractions[outInterIdx].insideIdxInBlock) == outsideInteractions[outInterIdx].insideIndex);
            FOpenCLAssertLF(particles.nbParticles);
            FOpenCLAssertLF(interParticles.nbParticles);

            P2POuter( GetPositionFromMorton(outsideInteractions[outInterIdx].insideIndex, treeHeight-1), particles, particles ,
                      interParticles, outsideInteractions[outInterIdx].relativeOutPosition, userkernel );

            P2POuter( GetPositionFromMorton(outsideInteractions[outInterIdx].outIndex, treeHeight-1), interParticles, interParticles ,
                      particles, FMGetOppositeNeighIndex(outsideInteractions[outInterIdx].relativeOutPosition), userkernel);
        }
    }
}



/////////////////////////////////////////////////////////////////////////////////////
/// Merge Pass
/////////////////////////////////////////////////////////////////////////////////////



__kernel void FOpenCL__mergePassPerform(__global unsigned char* leafCellsPtr, size_t leafCellsSize, __global unsigned char* leafCellsDownPtr,
                                        __global unsigned char* containersPtr, size_t containersSize, __global unsigned char* containersDownPtr,
                                        __global void* userkernel){
    struct FOpenCLGroupOfCells leafCells = BuildFOpenCLGroupOfCells(leafCellsPtr,leafCellsSize, NULLPTR, leafCellsDownPtr);
    struct FOpenCLGroupOfParticles containers = BuildFOpenCLGroupOfParticles(containersPtr,containersSize, containersDownPtr);

    const int nbLeaves = FOpenCLGroupOfCells_getNumberOfCellsInBlock(&leafCells);

    for(int idxLeaf = 0 ; idxLeaf < nbLeaves ; ++idxLeaf){
        struct FWrappeCell cell = FOpenCLGroupOfCells_getDownCell(&leafCells, idxLeaf);
        FOpenCLAssertLF(cell.symb->mortonIndex == FOpenCLGroupOfCells_getCellMortonIndex(&leafCells, idxLeaf));
        struct FOpenCLGroupAttachedLeaf particles = FOpenCLGroupOfParticles_getLeaf(&containers, idxLeaf);
        FOpenCLAssertLF(FOpenCLGroupOfParticles_getLeafMortonIndex(&containers, idxLeaf) == FOpenCLGroupOfCells_getCellMortonIndex(&leafCells, idxLeaf));
        L2P(cell, particles, userkernel);
    }
}

