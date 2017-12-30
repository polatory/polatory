// See LICENCE file at project root
#ifndef FOCTREE_HPP
#define FOCTREE_HPP

#include <functional>

#include "FSubOctree.hpp"
#include "FTreeCoordinate.hpp"
#include "FBlockAllocator.hpp"

#include "../Utils/FLog.hpp"
#include "../Utils/FGlobal.hpp"
#include "../Utils/FGlobalPeriodic.hpp"
#include "../Utils/FPoint.hpp"
#include "../Utils/FMath.hpp"
#include "../Utils/FNoCopyable.hpp"
#include "../Utils/FAssert.hpp"
#include "FCoordinateComputer.hpp"

/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class FOctree
 * Please read the license
 *
 * This class is an octree container.
 *
 * Please refere to testOctree.cpp to see an example.
 * @code
 * // can be used as : <br>
 * FOctree<TestParticle, TestCell> tree(1.0,FPoint<FReal>(0.5,0.5,0.5));
 * @endcode
 *
 * Particles and cells has to respect the Abstract class definition.
 * Particle must extend {FExtendPosition}
 * Cell must extend extend proposes accessors to FTreeCoordinate and MortonIndex.
 *
 * If the octree as an height H, then it goes from 0 to H-1
 * at level 0 the space is not split
 * CellAllocator can be FListBlockAllocator<CellClass, 10> or FBasicBlockAllocator<CellClass>
 */
template<class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass = FBasicBlockAllocator<CellClass> /*FListBlockAllocator<CellClass, 15>*/ >
class FOctree : public FNoCopyable {
public:
    using FRealType = FReal;
    using CellClassType = CellClass;
    using ContainerClassType = ContainerClass;
    using LeafClassType = LeafClass;                             //< The type of the Leaf used in the Octree

protected:
    typedef FOctree<FReal, CellClass , ContainerClass, LeafClass, CellAllocatorClass>      OctreeType;
    typedef  FSubOctreeWithLeafs<FReal, CellClass , ContainerClass, LeafClass, CellAllocatorClass> SubOctreeWithLeaves;
    typedef FSubOctree<FReal, CellClass , ContainerClass, LeafClass, CellAllocatorClass>           SubOctree;

    FAbstractSubOctree<FReal, CellClass , ContainerClass, LeafClass, CellAllocatorClass>* root;   //< root suboctree

    FReal*const boxWidthAtLevel;	//< to store the width of each boxs at all levels

    const int height;                   //< tree height
    const int subHeight;		//< tree height
    const int leafIndex;		//< index of leaf int array

    const FPoint<FReal> boxCenter;	//< the space system center
    const FPoint<FReal> boxCorner;	//< the space system corner (used to compute morton index)

    const FReal boxWidth;       //< the space system width


    /**
     * Get morton index from a position for the leaf level
     * @param inPosition position to compute
     * @return the morton index
     */
    FTreeCoordinate getCoordinateFromPosition(const FPoint<FReal>& inPosition) const {
        return FCoordinateComputer::GetCoordinateFromPositionAndCorner<FReal>(this->boxCorner, this->boxWidth, height, inPosition);
    }

    /**
     * Get the box number from a position
     * at a position POS with a leaf level box width of WI, the position is RELATIVE_TO_CORNER(POS)/WI
     * @param inRelativePosition a position from the corner of the box
     * @return the box num at the leaf level that contains inRelativePosition
     */
    int getTreeCoordinate(const FReal inRelativePosition) const {
        return FCoordinateComputer::GetTreeCoordinate<FReal>(inRelativePosition, this->boxWidth, this->boxWidthAtLevel[this->leafIndex], height);
    }

public:
    /**
     * Constructor
     * @param inHeight the octree height
     * @param inSubHeight the octree subheight
     * @param inBoxWidth box width for this simulation
     * @param inBoxCenter box center for this simulation
     */
    FOctree(const int inHeight, const int inSubHeight,
            const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter)
        : root(nullptr), boxWidthAtLevel(new FReal[inHeight]),
          height(inHeight) , subHeight(inSubHeight), leafIndex(this->height-1),
          boxCenter(inBoxCenter), boxCorner(inBoxCenter,-(inBoxWidth/2)), boxWidth(inBoxWidth)
    {
        FAssertLF(subHeight <= height - 1, "Subheight cannot be greater than height", __LINE__, __FILE__ );
        // Does we only need one suboctree?
        if(subHeight == height - 1){
            root = new FSubOctreeWithLeafs< FReal, CellClass , ContainerClass, LeafClass,CellAllocatorClass>(nullptr, 0, this->subHeight, 1);
        }
        else {// if(subHeight < height - 1)
            root = new FSubOctree< FReal, CellClass , ContainerClass, LeafClass,CellAllocatorClass>(nullptr, 0, this->subHeight, 1);
        }

        FReal tempWidth = this->boxWidth;
        // pre compute box width for each level
        for(int indexLevel = 0; indexLevel < this->height; ++indexLevel ){
            this->boxWidthAtLevel[indexLevel] = tempWidth;
            tempWidth /= FReal(2.0);
        }
    }

    /** Desctructor */
    virtual ~FOctree() {
        delete [] boxWidthAtLevel;
        delete root;
    }

    /** To get the tree height */
    int getHeight() const {
        return this->height;
    }

    /** To get the tree subheight */
    int getSubHeight() const{
        return this->subHeight;
    }

    /** To get the box width */
    FReal getBoxWidth() const{
        return this->boxWidth;
    }

    /** To get the center of the box */
    const FPoint<FReal>& getBoxCenter() const{
        return this->boxCenter;
    }

    /** Count the number of cells per level,
     * it will iter on the tree to do that!
     */
    void getNbCellsPerLevel(int inNbCells[]){
        Iterator octreeIterator(this);
        octreeIterator.gotoBottomLeft();

        Iterator avoidGoLeft(octreeIterator);

        for(int idxLevel = height - 1 ; idxLevel > 0; --idxLevel ){
            int counter = 0;
            do{
                ++counter;
            } while(octreeIterator.moveRight());
            avoidGoLeft.moveUp();
            octreeIterator = avoidGoLeft;
            inNbCells[idxLevel] = counter;
        }

        inNbCells[0] = 0;
    }

    /**
     * Insert a particle on the tree
     * algorithm is :
     * Compute morton index for the particle
     * ask node to insert this particle
     * @param inParticle the particle to insert (must inherit from FAbstractParticle)
     */
    template<typename... Args>
    void insert(const FPoint<FReal>& inParticlePosition, Args... args){
        const FTreeCoordinate host = getCoordinateFromPosition( inParticlePosition );
        const MortonIndex particleIndex = host.getMortonIndex();
        if(root->isLeafPart()){
            ((SubOctreeWithLeaves*)root)->insert( particleIndex, host, this->height, inParticlePosition, args... );
        }
        else{
            ((SubOctree*)root)->insert( particleIndex, host, this->height, inParticlePosition, args... );
        }
    }

    /** Remove a leaf from its morton index
     * @param indexToRemove the index of the leaf to remove
     */
    LeafClass* createLeaf(const MortonIndex indexToCreate ){
        const FTreeCoordinate host(indexToCreate);
        if(root->isLeafPart()){
            return ((SubOctreeWithLeaves*)root)->createLeaf( indexToCreate, host, this->height );
        }
        else{
            return ((SubOctree*)root)->createLeaf( indexToCreate, host, this->height );
        }
    }

    /** Remove a leaf from its morton index
     * @param indexToRemove the index of the leaf to remove
     */
    void removeLeaf(const MortonIndex indexToRemove ){
        root->removeLeaf( indexToRemove , this->height);
    }

    /**
     * Get a morton index from a real position
     * @param position a position to compute MI
     * @return the morton index
     */
    MortonIndex getMortonFromPosition(const FPoint<FReal>& position) const {
        return getCoordinateFromPosition(position).getMortonIndex();
    }

    /*
     * Indicate if tree is empty or not.
     */
    bool isEmpty(){
        return root->getRightLeafIndex() < 0;
    }

    /**
     * The class works on suboctree. Most of the resources needed
     * are avaiblable by using FAbstractSubOctree. But when accessing
     * to the leaf we have to use FSubOctree or FSubOctreeWithLeafs
     * depending if we are working on the bottom of the tree.
     */
    union SubOctreeTypes {
        FAbstractSubOctree<FReal,CellClass,ContainerClass,LeafClass,CellAllocatorClass>* tree;     //< Usual pointer to work
        FSubOctree<FReal,CellClass,ContainerClass,LeafClass,CellAllocatorClass>* middleTree;       //< To access to sub-octree under
        FSubOctreeWithLeafs<FReal,CellClass,ContainerClass,LeafClass,CellAllocatorClass>* leafTree;//< To access to particles lists
    };

    /**
     * This class is a const SubOctreeTypes
     */
    union SubOctreeTypesConst {
        const FAbstractSubOctree<FReal,CellClass,ContainerClass,LeafClass,CellAllocatorClass>* tree;     //< Usual pointer to work
        const FSubOctree<FReal,CellClass,ContainerClass,LeafClass,CellAllocatorClass>* middleTree;       //< To access to sub-octree under
        const FSubOctreeWithLeafs<FReal,CellClass,ContainerClass,LeafClass,CellAllocatorClass>* leafTree;//< To access to particles lists
    };

    /**
     * This has to be used to iterate on an octree
     * It simply stores an pointer on a suboctree and moves to right/left/up/down.
     * Please refer to testOctreeIter file to see how it works.
     *
     * @code
     * FOctree<TestParticle, TestCell, NbLevels, NbSubLevels>::Iterator octreeIterator(&tree);
     * octreeIterator.gotoBottomLeft();
     * for(int idx = 0 ; idx < NbLevels - 1; ++idx ){
     *     do{
     *         // ...
     *     } while(octreeIterator.moveRight());
     *     octreeIterator.moveUp();
     *     octreeIterator.gotoLeft();
     * }
     * @endcode
     * Remark :
     * It uses the left right limit on each suboctree and their morton index.
     * Please have a look to the move functions to understand how the system is working.
     */
    class Iterator  {
        SubOctreeTypes current; //< Current suboctree

        int currentLocalIndex;  //< Current index (array position) in the current_suboctree.cells[ currentLocalLevel ]
        int currentLocalLevel;  //< Current level in the current suboctree

        /**
         * To know what is the left limit on the current level on the current subtree
         * @return suboctree.left_limit >> 3 * diff(leaf_index, current_index).
         */
        static int TransposeIndex(const int indexInLeafLevel, const int distanceFromLeafLevel) {
            return indexInLeafLevel >> (3 * distanceFromLeafLevel);
        }


    public:
        /**
         * Constructor
         * @param inTarget the octree to iterate on
         * After building a iterator, this one is positioned at the level 0
         * of the root (level 1 of octree) at the left limit index
         */
        explicit Iterator(FOctree* const inTarget)
            : currentLocalIndex(0) , currentLocalLevel(0) {
            FAssertLF(inTarget, "Target for Octree::Iterator cannot be null", __LINE__, __FILE__);
            FAssertLF(inTarget->root->getRightLeafIndex() >= 0, "Octree seems to be empty, getRightLeafIndex == 0", __LINE__, __FILE__);

            // Start by the root
            this->current.tree = inTarget->root;
            // On the left limit
            this->currentLocalIndex = TransposeIndex(this->current.tree->getLeftLeafIndex(), (this->current.tree->getSubOctreeHeight() - this->currentLocalLevel - 1) );
        }

        Iterator() : currentLocalIndex(0),currentLocalLevel(0) {
            current.tree = nullptr;
        }

        /** Copy constructor
         * @param other source iterator to copy
         */
        Iterator(const Iterator& other){
            this->current = other.current ;
            this->currentLocalLevel = other.currentLocalLevel ;
            this->currentLocalIndex = other.currentLocalIndex ;
        }

        /** Move constructor
         * @param other source iterator to move.
         */
        Iterator(Iterator&& other){
            this->current = other.current ;
            this->currentLocalLevel = other.currentLocalLevel ;
            this->currentLocalIndex = other.currentLocalIndex ;
        }

        /** Copy operator
         * @param other source iterator to copy
         * @return this after copy
         */
        Iterator& operator=(const Iterator& other){
            this->current = other.current ;
            this->currentLocalLevel = other.currentLocalLevel ;
            this->currentLocalIndex = other.currentLocalIndex ;
            return *this;
        }

        /** \brief Move operator
         * \param other Object to move.
         * \return This object after move.
         */
        Iterator& operator=(Iterator&& other) {
            this->current = other.current ;
            this->currentLocalLevel = other.currentLocalLevel ;
            this->currentLocalIndex = other.currentLocalIndex ;
            return *this;
        }

        /** \brief equality operator
         *
         * \param other Iterator to compare to.
         * \return True if the two oterators are the same, false otherwise.
         */
        bool operator==(const Iterator& other) const {
            // Note C++11 standard, $5.9.4:
            // If two pointers point to non-static data members of the same
            // union object, they compare equal (after conversion to void*, if
            // necessary). If two pointers point to elements of the same array
            // or one beyond the end of the array, the pointer to the object
            // with the higher subscript compares higher.
            //
            // This is why we compare the 'tree' component of the 'current' union
            return this->current.tree == other.current.tree &&
                    this->currentLocalLevel == other.currentLocalLevel &&
                    this->currentLocalIndex == other.currentLocalIndex ;
        }

        /** \brief Inequality operator
         * \param other Iterator to compare to.
         * \return True if the two oterators are the different, false otherwise.
         */
        bool operator!=(const Iterator& other) const {
            return ! (*this == other);
        }

        /**
         * Move iterator to the top! (level 0 of root suboctree, level 1 of octree)
         * after this function : index = left limit at root level
         * the Algorithm is :
         *     going to root suboctree
         *     going to the first level and most left node
         */
        void gotoTop(){
            while(this->current.tree->hasParent()){
                this->current.tree = this->current.tree->getParent();
            }
            this->currentLocalLevel = 0;
            this->currentLocalIndex = TransposeIndex(this->current.tree->getLeftLeafIndex(), (this->current.tree->getSubOctreeHeight() - 1) );
        }

        /**
         * Move iterator to the bottom left place
         * We are on a leaf a the most left node
         * the Algorithm is :
         *     first go to top
         *     then stay on the left and go downward
         */
        void gotoBottomLeft(){
            gotoTop();
            while(1) {
                this->currentLocalLevel = this->current.tree->getSubOctreeHeight() - 1;
                this->currentLocalIndex = this->current.tree->getLeftLeafIndex();
                if( isAtLeafLevel() ){
                    return;
                }
                this->current.tree = this->current.middleTree->leafs( this->currentLocalIndex );
            }
        }

        /**
         * Move iterator to the left place at the same level
         * if needed we go on another suboctree but we stay on at the same level
         * the Algorithm is :
         *     go to top
         *     go downward until we are a the same level
         */
        void gotoLeft(){
            //  Function variables
            const int currentLevel = level();

            // Goto root sutoctree
            while( this->current.tree->hasParent() ){
                this->current.tree = this->current.tree->getParent();
            }

            // Go down on the left until arriving on the same global level
            while( this->current.tree->getSubOctreeHeight() + this->current.tree->getSubOctreePosition() - 1 < currentLevel ) {
                this->current.tree = this->current.middleTree->leafs(this->current.tree->getLeftLeafIndex());
            }

            // Level still unchanged we only go to the left
            // The left limit on this tree at the level we want to stay
            this->currentLocalIndex = TransposeIndex(this->current.tree->getLeftLeafIndex(), (this->current.tree->getSubOctreeHeight() - this->currentLocalLevel - 1));
        }

        /**
         * Move iterator to the right place at the same level
         * if needed we go on another suboctree but we stay on at the same level
         * the Algorithm is :
         *     go to top
         *     go downward until we are a the same level
         */
        void gotoRight(){
            //  Function variables
            const int currentLevel = level();
            // Goto root sutoctree
            while( this->current.tree->hasParent() ){
                this->current.tree = this->current.tree->getParent();
            }
            // Go down on the left until arriving on the same global level
            while( this->current.tree->getSubOctreeHeight() + this->current.tree->getSubOctreePosition() - 1 < currentLevel ) {
                this->current.tree = this->current.middleTree->leafs(this->current.tree->getRightLeafIndex());
            }
            // Level still unchanged we only go to the left
            // The left limit on this tree at the level we want to stay
            this->currentLocalIndex = TransposeIndex(this->current.tree->getRightLeafIndex(), (this->current.tree->getSubOctreeHeight() - this->currentLocalLevel - 1));
        }

        /**
         * Goto the next value on the right at the same level
         *
         * The algorithm here is :
         * As long as we are on the right limit, go to the parent suboctree
         * if we are on the root and on the right then return (there is no more data on the right)
         *
         * After that point we do not know where we are but we know that there is some data
         * on the right (without knowing our position!)
         *
         * We gotoNext on the brother to find an allocated cell (->)
         * for example if we are on index 2 we will look until 8 = 2 | 7 + 1
         * if we arrive a 8 without finding a cell we go upper and do the same
         * we know we will find something because we are not at the right limit
         *
         * We find an allocated cell.
         * We have to go down, we go on the left child of this cells
         * until : the current level if we did not have change the current suboctree
         * or : the leaf level
         *
         * In the second case, it means we need to change octree downward
         * but it is easy because we can use the left limit!
         *
         * @return true if we succeed to go to the right, else false
         */
        bool moveRight(){
            //  Function variables
            SubOctreeTypes workingTree = this->current;    // To cover parent other sub octre
            int workingLevel = this->currentLocalLevel;        // To know where we are
            int workingIndex = this->currentLocalIndex;        // To know where we are

            // -- First we have to go in a tree where we can move on the right --
            // Number of time we go into parent subtree
            int countUpward = 0;
            // We stop when we can right move or if there is no more parent (root)
            while( workingIndex == TransposeIndex(workingTree.tree->getRightLeafIndex(), (workingTree.tree->getSubOctreeHeight() - workingLevel - 1) )
                   && workingTree.tree->hasParent() ){
                // Goto the leaf level into parent at current_tree.position_into_parent_array
                workingIndex        = workingTree.tree->getIndexInParent();
                workingTree.tree    = workingTree.tree->getParent();
                workingLevel        = workingTree.tree->getSubOctreeHeight() - 1;
                // inc counter
                ++countUpward;
            }

            // Do we stop because we are on the root (means we cannot move right?)
            if( workingIndex < TransposeIndex(workingTree.tree->getRightLeafIndex(), (workingTree.tree->getSubOctreeHeight() - workingLevel - 1) ) ){
                // Move to the first right cell pointer(!)
                ++workingIndex;

                // Maybe it is null, but we know there is almost one cell on the right
                // we need to find it
                if( !workingTree.tree->cellsAt(workingLevel)[workingIndex] ){
                    // While we are not on a allocated cell
                    while( true ){
                        // Test element on the right (test brothers)
                        const int rightLimite = (workingIndex | 7) + 1;
                        while( workingIndex < rightLimite && !workingTree.tree->cellsAt(workingLevel)[workingIndex]){
                            ++workingIndex;
                        }
                        // Stop if we are on an allocated cell
                        if( workingTree.tree->cellsAt(workingLevel)[workingIndex] ){
                            break;
                        }
                        // Else go to the upper level
                        --workingLevel;
                        workingIndex >>= 3;
                    }
                }

                // if wokring tree != current tree => working tree leafs level ; else current level
                const int objectiveLevel = (countUpward ? workingTree.tree->getSubOctreeHeight() - 1 : this->currentLocalLevel );

                // We need to go down as left as possible
                while( workingLevel != objectiveLevel ){
                    ++workingLevel;
                    workingIndex <<= 3;
                    const int rightLimite = (workingIndex | 7); // not + 1 because if the 7th first are null it must be the 8th!
                    while( workingIndex < rightLimite && !workingTree.tree->cellsAt(workingLevel)[workingIndex]){
                        ++workingIndex;
                    }
                }

                // Do we change from the current sub octree?
                if( countUpward ){
                    // Then we simply need to go down the same number of time
                    workingTree.tree = workingTree.middleTree->leafs(workingIndex);
                    while( --countUpward ){
                        workingTree.tree = workingTree.middleTree->leafs(workingTree.tree->getLeftLeafIndex());
                    }
                    // level did not change, simpli set octree and get left limit of this octree at the current level
                    this->current = workingTree;
                    this->currentLocalIndex = TransposeIndex(workingTree.tree->getLeftLeafIndex(), (workingTree.tree->getSubOctreeHeight() - this->currentLocalLevel - 1) );
                }
                else{
                    // We are on the right tree
                    this->currentLocalIndex = workingIndex;
                }

                return true;
            }
            return false;
        }

        /**
         * Move to the upper level
         * It may cause to change the suboctree we are working on
         * but we are using the same morton index >> 3
         * @return true if succeed
         */
        bool moveUp() {
            // It is on the top level?
            if( this->currentLocalLevel ){
                // No so simply go up
                --this->currentLocalLevel;
                this->currentLocalIndex >>= 3;
            }
            // Yes need to change suboctree
            else if( this->current.tree->hasParent() ){
                this->currentLocalIndex = this->current.tree->getIndexInParent();
                this->current.tree = this->current.tree->getParent();
                this->currentLocalLevel =  this->current.tree->getSubOctreeHeight() - 1;
            }
            else{
                return false;
            }
            return true;
        }

        /**
         * Move down
         * It may cause to change the suboctree we are working on
         * We point on the first child found from left to right in the above
         * level
         * @return true if succeed
         */
        bool moveDown(){
            if( !isAtLeafLevel() ){
                // We are on the leaf of the current suboctree?
                if(this->currentLocalLevel + 1 == this->current.tree->getSubOctreeHeight()){
                    // Yes change sub octree
                    this->current.tree = this->current.middleTree->leafs(this->currentLocalIndex);
                    this->currentLocalIndex = 0;
                    this->currentLocalLevel = 0;
                }
                // No simply go down
                else{
                    ++this->currentLocalLevel;
                    this->currentLocalIndex <<= 3;
                }
                // Find the first allocated cell from left to right
                while(!this->current.tree->cellsAt(this->currentLocalLevel)[this->currentLocalIndex]){
                    ++this->currentLocalIndex;
                }
                return true;
            }
            return false;
        }

        /**
         * To know if we are not on the root level
         * @return true if we can move up
         */
        bool canProgressToUp() const {
            return this->currentLocalLevel || this->current.tree->hasParent();
        }

        /**
         * To know if we are not on the leafs level
         * @return true if we can move down
         */
        bool canProgressToDown() const {
            return !isAtLeafLevel();
        }

        /**
         * To know if we are on the leafs level
         * @return true if we are at the bottom of the tree
         */
        bool isAtLeafLevel() const {
            return this->current.tree->isLeafPart() && this->currentLocalLevel + 1 == this->current.tree->getSubOctreeHeight();
        }

        /**
         * To know the current level (not local but global)
         * @return the level in the entire octree
         */
        int level() const {
            return this->currentLocalLevel + this->current.tree->getSubOctreePosition();
        }

        /** Get the current pointed leaf
         * @return current leaf element
         */
        LeafClass* getCurrentLeaf() const {
            return this->current.leafTree->getLeaf(this->currentLocalIndex);
        }

        /** To access the current particles list
         * You have to be at the leaf level to call this function!
         * @return current element list
         */
        ContainerClass* getCurrentListSrc() const {
            return this->current.leafTree->getLeafSrc(this->currentLocalIndex);
        }

        /** To access the current particles list
         * You have to be at the leaf level to call this function!
         * @return current element list
         */
        ContainerClass* getCurrentListTargets() const {
            return this->current.leafTree->getLeafTargets(this->currentLocalIndex);
        }

        /** Get the current pointed cell
         * @return current cell element
         */
        CellClass* getCurrentCell() const {
            return this->current.tree->cellsAt(this->currentLocalLevel)[this->currentLocalIndex];
        }

        /** Gets the children of the current cell.
         *
         * This function return an array of 8 CellClass. To know whether
         * a child cell exists or not, the pointer must be checked.
         *
         * @return the 8-child array.
         */
        CellClass** getCurrentChild() const {
            // are we at the bottom of the suboctree
            if(this->current.tree->getSubOctreeHeight() - 1 == this->currentLocalLevel ){
                // then return first level of the suboctree under
                return &this->current.middleTree->leafs(this->currentLocalIndex)->cellsAt(0)[0];
            } else {
                // else simply return the array at the right position
                return &this->current.tree->cellsAt(this->currentLocalLevel + 1)[this->currentLocalIndex << 3];
            }
        }

        /** Gets the children of the current cell.
         *
         * This function return an array of 8 CellClass. To know whether
         * a child cell exists or not, the pointer must be checked.
         *
         * @return the 8-child array.
         */
        CellClass** getCurrentChildren() const {
            return getCurrentChild();
        }

        /** Get the part of array that contains all the pointers
         *
         */
        CellClass** getCurrentBox() const {
            return &this->current.tree->cellsAt(this->currentLocalLevel)[this->currentLocalIndex & ~7];
        }

        /** Get the Morton index of the current cell pointed by the iterator
         * @return The global morton index
         * <code>iter.getCurrentGlobalIndex();<br>
         * // is equivalent to :<br>
         * iter.getCurrentCell()->getMortonIndex();</code>
         */
        MortonIndex getCurrentGlobalIndex() const{
            return this->current.tree->cellsAt(this->currentLocalLevel)[this->currentLocalIndex]->getMortonIndex();
        }

        /** To get the tree coordinate of the current working cell
         *
         */
        const FTreeCoordinate& getCurrentGlobalCoordinate() const{
            return this->current.tree->cellsAt(this->currentLocalLevel)[this->currentLocalIndex]->getCoordinate();
        }

    };

    // To be able to access octree root & data
    friend class Iterator;

    ///////////////////////////////////////////////////////////////////////////
    // This part is related to the FMM algorithm (needed by M2M,M2L,etc.)
    ///////////////////////////////////////////////////////////////////////////

    /** This function return a cell (if it exists) from a morton index and a level
     * @param inIndex the index of the desired cell
     * @param inLevel the level of the desired cell (cannot be inferred from the index)
     * @return the cell if it exist or null (0)
     * This function starts from the root until it find a missing cell or the right cell
     */
    CellClass* getCell(const MortonIndex inIndex, const int inLevel) const{
        SubOctreeTypesConst workingTree;
        workingTree.tree = this->root;
        const MortonIndex treeSubLeafMask = ~(~0x00LL << (3 *  workingTree.tree->getSubOctreeHeight() ));

        // Find the suboctree a the correct level
        while(inLevel >= workingTree.tree->getSubOctreeHeight() + workingTree.tree->getSubOctreePosition()) {
            // compute the leaf index
            const MortonIndex fullIndex = inIndex >> (3 * (inLevel + 1 - (workingTree.tree->getSubOctreeHeight() + workingTree.tree->getSubOctreePosition()) ) );
            // point to next suboctree
            workingTree.tree = workingTree.middleTree->leafs(treeSubLeafMask & fullIndex);
            if(!workingTree.tree) return nullptr;
        }

        // compute correct index in the array
        const MortonIndex treeLeafMask = ~(~0x00LL << (3 *  (inLevel + 1 - workingTree.tree->getSubOctreePosition()) ));
        return workingTree.tree->cellsAt(inLevel - workingTree.tree->getSubOctreePosition())[treeLeafMask & inIndex];
    }


    /** This function fill a array with the neighbors of a cell
     * it does not put the brothers in the array (brothers are cells
     * at the same level with the same parent) because they are of course
     * direct neighbors.
     * There is a maximum of 26 (3*3*3-1) direct neighbors
     *  // Take the neighbors != brothers
     *  CellClass* directNeighbors[26];
     *  const int nbDirectNeighbors = getNeighborsNoBrothers(directNeighbors,inIndex,inLevel);
     * @param inNeighbors the array to store the elements
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getNeighborsNoBrothers(CellClass* inNeighbors[26], const MortonIndex inIndex, const int inLevel) const {
        FTreeCoordinate center;
        center.setPositionFromMorton(inIndex);

        const int boxLimite = FMath::pow2(inLevel);

        int idxNeighbors = 0;

        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(center.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(center.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(center.getZ() + idxZ,0,boxLimite)) continue;

                    // if we are not on the current cell
                    if( !(!idxX && !idxY && !idxZ) ){
                        const FTreeCoordinate other(center.getX() + idxX,center.getY() + idxY,center.getZ() + idxZ);
                        const MortonIndex mortonOther = other.getMortonIndex();
                        // if not a brother
                        if( mortonOther>>3 != inIndex>>3 ){
                            // get cell
                            CellClass* const cell = getCell(mortonOther, inLevel);
                            // add to list if not null
                            if(cell) inNeighbors[idxNeighbors++] = cell;
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }


    /** This function return an address of cell array from a morton index and a level
     *
     * @param inIndex the index of the desired cell array has to contains
     * @param inLevel the level of the desired cell (cannot be inferred from the index)
     * @return the cell if it exist or null (0)
     *
     */
    CellClass** getCellPt(const MortonIndex inIndex, const int inLevel) const{
        SubOctreeTypesConst workingTree;
        workingTree.tree = this->root;

        const MortonIndex treeMiddleMask = ~(~0x00LL << (3 *  workingTree.tree->getSubOctreeHeight() ));

        // Find the suboctree a the correct level
        while(inLevel >= workingTree.tree->getSubOctreeHeight() + workingTree.tree->getSubOctreePosition()) {
            // compute the leaf index
            const MortonIndex fullIndex = inIndex >> 3 * (inLevel + 1 - (workingTree.tree->getSubOctreeHeight() + workingTree.tree->getSubOctreePosition()) );
            // point to next suboctree
            workingTree.tree = workingTree.middleTree->leafs(int(treeMiddleMask & fullIndex));
            if(!workingTree.tree) return nullptr;
        }

        // Be sure there is a parent allocated
        const int levelInTree = inLevel - workingTree.tree->getSubOctreePosition();
        if( levelInTree && !workingTree.tree->cellsAt(levelInTree - 1)[~(~0x00LL << (3 * levelInTree )) & (inIndex>>3)]){
            return nullptr;
        }

        // compute correct index in the array and return the @ in array
        const MortonIndex treeLeafMask = ~(~0x00LL << (3 * (levelInTree + 1 ) ));
        return &workingTree.tree->cellsAt(levelInTree)[treeLeafMask & inIndex];
    }


    /** This function fill an array with the distant neighbors of a cell
     * @param inNeighbors the array to store the elements
     * @param inNeighborsIndex the array to store morton index of the neighbors
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getInteractionNeighbors(const CellClass* inNeighbors[343],
    const FTreeCoordinate& workingCell,
    const int inLevel, const int neighSeparation = 1) const{
        // reset
        memset(inNeighbors, 0, sizeof(CellClass*) * 343);

        // Then take each child of the parent's neighbors if not in directNeighbors
        // Father coordinate
        const FTreeCoordinate parentCell(workingCell.getX()>>1,workingCell.getY()>>1,workingCell.getZ()>>1);

        // Limite at parent level number of box (split by 2 by level)
        const int boxLimite = FMath::pow2(inLevel-1);

        int idxNeighbors = 0;
        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(parentCell.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(parentCell.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(parentCell.getZ() + idxZ,0,boxLimite)) continue;

                    // if we are not on the current cell
                    if( neighSeparation<1 || idxX || idxY || idxZ ){
                        const FTreeCoordinate otherParent(parentCell.getX() + idxX,parentCell.getY() + idxY,parentCell.getZ() + idxZ);
                        const MortonIndex mortonOtherParent = otherParent.getMortonIndex() << 3;
                        // Get child
                        CellClass** const cells = getCellPt(mortonOtherParent, inLevel);

                        // If there is one or more child
                        if(cells){
                            // For each child
                            for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                                if(cells[idxCousin]){
                                    const int xdiff  = ((otherParent.getX()<<1) | ( (idxCousin>>2) & 1)) - workingCell.getX();
                                    const int ydiff  = ((otherParent.getY()<<1) | ( (idxCousin>>1) & 1)) - workingCell.getY();
                                    const int zdiff  = ((otherParent.getZ()<<1) | (idxCousin&1)) - workingCell.getZ();

                                    // Test if it is a direct neighbor
                                    if(FMath::Abs(xdiff) > neighSeparation || FMath::Abs(ydiff) > neighSeparation || FMath::Abs(zdiff) > neighSeparation){
                                        // add to neighbors
                                        inNeighbors[ (((xdiff+3) * 7) + (ydiff+3)) * 7 + zdiff + 3] = cells[idxCousin];
                                        ++idxNeighbors;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }

    /** This function fill an array with the distant neighbors of a cell
     * @param inNeighbors the array to store the elements
     * @param inNeighborsIndex the array to store morton index of the neighbors
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getInteractionNeighbors(const CellClass* inNeighbors[342],
    int inNeighborPositions[343],
    const FTreeCoordinate& workingCell,
    const int inLevel, const int neighSeparation = 1) const{
        // Then take each child of the parent's neighbors if not in directNeighbors
        // Father coordinate
        const FTreeCoordinate parentCell(workingCell.getX()>>1,workingCell.getY()>>1,workingCell.getZ()>>1);

        // Limite at parent level number of box (split by 2 by level)
        const int boxLimite = FMath::pow2(inLevel-1);

        int idxNeighbors = 0;
        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(parentCell.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(parentCell.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(parentCell.getZ() + idxZ,0,boxLimite)) continue;

                    // if we are not on the current cell
                    if( neighSeparation<1 || idxX || idxY || idxZ ){
                        const FTreeCoordinate otherParent(parentCell.getX() + idxX,parentCell.getY() + idxY,parentCell.getZ() + idxZ);
                        const MortonIndex mortonOtherParent = otherParent.getMortonIndex() << 3;
                        // Get child
                        CellClass** const cells = getCellPt(mortonOtherParent, inLevel);

                        // If there is one or more child
                        if(cells){
                            // For each child
                            for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                                if(cells[idxCousin]){
                                    const int xdiff  = ((otherParent.getX()<<1) | ( (idxCousin>>2) & 1)) - workingCell.getX();
                                    const int ydiff  = ((otherParent.getY()<<1) | ( (idxCousin>>1) & 1)) - workingCell.getY();
                                    const int zdiff  = ((otherParent.getZ()<<1) | (idxCousin&1)) - workingCell.getZ();

                                    // Test if it is a direct neighbor
                                    if(FMath::Abs(xdiff) > neighSeparation || FMath::Abs(ydiff) > neighSeparation || FMath::Abs(zdiff) > neighSeparation){
                                        // add to neighbors
                                        inNeighbors[idxNeighbors] = cells[idxCousin];
                                        inNeighborPositions[idxNeighbors] = (((xdiff+3) * 7) + (ydiff+3)) * 7 + zdiff + 3;
                                        ++idxNeighbors;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }

    /** This function fills an array with all the neighbors of a cell,
     * i.e. Child of parent's neighbors, direct neighbors and cell itself.
     * This is called for instance when the nearfield also needs to be approximated
     * in that cas we only call this function at the leaf level.
     * @param inNeighbors the array to store the elements
     * @param inNeighborsIndex the array to store morton index of the neighbors
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getFullNeighborhood(const CellClass* inNeighbors[343],
    const FTreeCoordinate& workingCell,
    const int inLevel) const{
        // reset
        memset(inNeighbors, 0, sizeof(CellClass*) * 343);

        // Then take each child of the parent's neighbors
        // Father coordinate
        const FTreeCoordinate parentCell(workingCell.getX()>>1,workingCell.getY()>>1,workingCell.getZ()>>1);

        // Limite at parent level number of box (split by 2 by level)
        const int boxLimite = FMath::pow2(inLevel-1);

        int idxNeighbors = 0;
        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(parentCell.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(parentCell.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(parentCell.getZ() + idxZ,0,boxLimite)) continue;

                    const FTreeCoordinate otherParent(parentCell.getX() + idxX,parentCell.getY() + idxY,parentCell.getZ() + idxZ);
                    const MortonIndex mortonOtherParent = otherParent.getMortonIndex() << 3;
                    // Get child
                    CellClass** const cells = getCellPt(mortonOtherParent, inLevel);

                    // If there is one or more child
                    if(cells){
                        // For each child
                        for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                            if(cells[idxCousin]){
                                const int xdiff  = ((otherParent.getX()<<1) | ( (idxCousin>>2) & 1)) - workingCell.getX();
                                const int ydiff  = ((otherParent.getY()<<1) | ( (idxCousin>>1) & 1)) - workingCell.getY();
                                const int zdiff  = ((otherParent.getZ()<<1) | (idxCousin&1)) - workingCell.getZ();

                                // add to neighbors
                                inNeighbors[ (((xdiff+3) * 7) + (ydiff+3)) * 7 + zdiff + 3] = cells[idxCousin];
                                ++idxNeighbors;
                            }
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }

    /** This function fills an array with all the neighbors of a cell,
     * i.e. Child of parent's neighbors, direct neighbors and cell itself.
     * This is called for instance when the nearfield also needs to be approximated
     * in that cas we only call this function at the leaf level.
     * @param inNeighbors the array to store the elements
     * @param inNeighborsIndex the array to store morton index of the neighbors
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getFullNeighborhood(const CellClass* inNeighbors[342],
    int inNeighborPositions[342],
    const FTreeCoordinate& workingCell,
    const int inLevel) const{
        // reset
        // Then take each child of the parent's neighbors
        // Father coordinate
        const FTreeCoordinate parentCell(workingCell.getX()>>1,workingCell.getY()>>1,workingCell.getZ()>>1);

        // Limite at parent level number of box (split by 2 by level)
        const int boxLimite = FMath::pow2(inLevel-1);

        int idxNeighbors = 0;
        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(parentCell.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(parentCell.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(parentCell.getZ() + idxZ,0,boxLimite)) continue;

                    const FTreeCoordinate otherParent(parentCell.getX() + idxX,parentCell.getY() + idxY,parentCell.getZ() + idxZ);
                    const MortonIndex mortonOtherParent = otherParent.getMortonIndex() << 3;
                    // Get child
                    CellClass** const cells = getCellPt(mortonOtherParent, inLevel);

                    // If there is one or more child
                    if(cells){
                        // For each child
                        for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                            if(cells[idxCousin]){
                                const int xdiff  = ((otherParent.getX()<<1) | ( (idxCousin>>2) & 1)) - workingCell.getX();
                                const int ydiff  = ((otherParent.getY()<<1) | ( (idxCousin>>1) & 1)) - workingCell.getY();
                                const int zdiff  = ((otherParent.getZ()<<1) | (idxCousin&1)) - workingCell.getZ();

                                // add to neighbors
                                inNeighbors[idxNeighbors] = cells[idxCousin];
                                inNeighborPositions[idxNeighbors] = (((xdiff+3) * 7) + (ydiff+3)) * 7 + zdiff + 3;
                                ++idxNeighbors;
                            }
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }

    /** This function fill an array with the distant neighbors of a cell
     * it respects the periodic condition and will give the relative distance
     * between the working cell and the neighbors
     * @param inNeighbors the array to store the elements
     * @param inRelativePosition the array to store the relative position of the neighbors
     * @param workingCell the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getPeriodicInteractionNeighbors(const CellClass* inNeighbors[343],
    const FTreeCoordinate& workingCell,
    const int inLevel, const int inDirection, const int neighSeparation = 1) const{

        // Then take each child of the parent's neighbors if not in directNeighbors
        // Father coordinate
        const FTreeCoordinate parentCell(workingCell.getX()>>1,workingCell.getY()>>1,workingCell.getZ()>>1);

        // Limite at parent level number of box (split by 2 by level)
        const int boxLimite = FMath::pow2(inLevel-1);

        // This is not on a border we can use normal interaction list method
        if( !(parentCell.getX() == 0 || parentCell.getY() == 0 || parentCell.getZ() == 0 ||
              parentCell.getX() == boxLimite - 1 || parentCell.getY() == boxLimite - 1 || parentCell.getZ() == boxLimite - 1 ) ) {
            return getInteractionNeighbors( inNeighbors, workingCell, inLevel);
        }
        else{
            // reset

            memset(inNeighbors, 0, sizeof(CellClass*) * 343);

            const int startX =  (TestPeriodicCondition(inDirection, DirMinusX) || parentCell.getX() != 0 ?-1:0);
            const int endX =    (TestPeriodicCondition(inDirection, DirPlusX)  || parentCell.getX() != boxLimite - 1 ?1:0);
            const int startY =  (TestPeriodicCondition(inDirection, DirMinusY) || parentCell.getY() != 0 ?-1:0);
            const int endY =    (TestPeriodicCondition(inDirection, DirPlusY)  || parentCell.getY() != boxLimite - 1 ?1:0);
            const int startZ =  (TestPeriodicCondition(inDirection, DirMinusZ) || parentCell.getZ() != 0 ?-1:0);
            const int endZ =    (TestPeriodicCondition(inDirection, DirPlusZ)  || parentCell.getZ() != boxLimite - 1 ?1:0);

            int idxNeighbors = 0;
            // We test all cells around
            for(int idxX = startX ; idxX <= endX ; ++idxX){
                for(int idxY = startY ; idxY <= endY ; ++idxY){
                    for(int idxZ = startZ ; idxZ <= endZ ; ++idxZ){
                        // if we are not on the current cell
                        if(neighSeparation<1 || idxX || idxY || idxZ ){

                            const FTreeCoordinate otherParent(parentCell.getX() + idxX,parentCell.getY() + idxY,parentCell.getZ() + idxZ);
                            FTreeCoordinate otherParentInBox(otherParent);
                            // periodic
                            if( otherParentInBox.getX() < 0 ){
                                otherParentInBox.setX( otherParentInBox.getX() + boxLimite );
                            }
                            else if( boxLimite <= otherParentInBox.getX() ){
                                otherParentInBox.setX( otherParentInBox.getX() - boxLimite );
                            }

                            if( otherParentInBox.getY() < 0 ){
                                otherParentInBox.setY( otherParentInBox.getY() + boxLimite );
                            }
                            else if( boxLimite <= otherParentInBox.getY() ){
                                otherParentInBox.setY( otherParentInBox.getY() - boxLimite );
                            }

                            if( otherParentInBox.getZ() < 0 ){
                                otherParentInBox.setZ( otherParentInBox.getZ() + boxLimite );
                            }
                            else if( boxLimite <= otherParentInBox.getZ() ){
                                otherParentInBox.setZ( otherParentInBox.getZ() - boxLimite );
                            }


                            const MortonIndex mortonOtherParent = otherParentInBox.getMortonIndex() << 3;
                            // Get child
                            CellClass** const cells = getCellPt(mortonOtherParent, inLevel);

                            // If there is one or more child
                            if(cells){
                                // For each child
                                for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                                    if(cells[idxCousin]){
                                        const int xdiff  = ((otherParent.getX()<<1) | ( (idxCousin>>2) & 1)) - workingCell.getX();
                                        const int ydiff  = ((otherParent.getY()<<1) | ( (idxCousin>>1) & 1)) - workingCell.getY();
                                        const int zdiff  = ((otherParent.getZ()<<1) | (idxCousin&1))         - workingCell.getZ();

                                        // Test if it is a direct neighbor
                                        if(FMath::Abs(xdiff) > neighSeparation || FMath::Abs(ydiff) > neighSeparation || FMath::Abs(zdiff) > neighSeparation){
                                            // add to neighbors
                                            inNeighbors[ (((xdiff+3) * 7) + (ydiff+3)) * 7 + zdiff + 3] = cells[idxCousin];
                                            ++idxNeighbors;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return idxNeighbors;
        }
    }

    /** This function fill an array with the distant neighbors of a cell
     * it respects the periodic condition and will give the relative distance
     * between the working cell and the neighbors
     * @param inNeighbors the array to store the elements
     * @param inRelativePosition the array to store the relative position of the neighbors
     * @param workingCell the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getPeriodicInteractionNeighbors(const CellClass* inNeighbors[342], int inNeighborPositions[342],
    const FTreeCoordinate& workingCell,
    const int inLevel, const int inDirection, const int neighSeparation = 1) const{

        // Then take each child of the parent's neighbors if not in directNeighbors
        // Father coordinate
        const FTreeCoordinate parentCell(workingCell.getX()>>1,workingCell.getY()>>1,workingCell.getZ()>>1);

        // Limite at parent level number of box (split by 2 by level)
        const int boxLimite = FMath::pow2(inLevel-1);

        // This is not on a border we can use normal interaction list method
        if( !(parentCell.getX() == 0 || parentCell.getY() == 0 || parentCell.getZ() == 0 ||
              parentCell.getX() == boxLimite - 1 || parentCell.getY() == boxLimite - 1 || parentCell.getZ() == boxLimite - 1 ) ) {
            return getInteractionNeighbors( inNeighbors, inNeighborPositions, workingCell, inLevel);
        }
        else{
            const int startX =  (TestPeriodicCondition(inDirection, DirMinusX) || parentCell.getX() != 0 ?-1:0);
            const int endX =    (TestPeriodicCondition(inDirection, DirPlusX)  || parentCell.getX() != boxLimite - 1 ?1:0);
            const int startY =  (TestPeriodicCondition(inDirection, DirMinusY) || parentCell.getY() != 0 ?-1:0);
            const int endY =    (TestPeriodicCondition(inDirection, DirPlusY)  || parentCell.getY() != boxLimite - 1 ?1:0);
            const int startZ =  (TestPeriodicCondition(inDirection, DirMinusZ) || parentCell.getZ() != 0 ?-1:0);
            const int endZ =    (TestPeriodicCondition(inDirection, DirPlusZ)  || parentCell.getZ() != boxLimite - 1 ?1:0);

            int idxNeighbors = 0;
            // We test all cells around
            for(int idxX = startX ; idxX <= endX ; ++idxX){
                for(int idxY = startY ; idxY <= endY ; ++idxY){
                    for(int idxZ = startZ ; idxZ <= endZ ; ++idxZ){
                        // if we are not on the current cell
                        if(neighSeparation<1 || idxX || idxY || idxZ ){

                            const FTreeCoordinate otherParent(parentCell.getX() + idxX,parentCell.getY() + idxY,parentCell.getZ() + idxZ);
                            FTreeCoordinate otherParentInBox(otherParent);
                            // periodic
                            if( otherParentInBox.getX() < 0 ){
                                otherParentInBox.setX( otherParentInBox.getX() + boxLimite );
                            }
                            else if( boxLimite <= otherParentInBox.getX() ){
                                otherParentInBox.setX( otherParentInBox.getX() - boxLimite );
                            }

                            if( otherParentInBox.getY() < 0 ){
                                otherParentInBox.setY( otherParentInBox.getY() + boxLimite );
                            }
                            else if( boxLimite <= otherParentInBox.getY() ){
                                otherParentInBox.setY( otherParentInBox.getY() - boxLimite );
                            }

                            if( otherParentInBox.getZ() < 0 ){
                                otherParentInBox.setZ( otherParentInBox.getZ() + boxLimite );
                            }
                            else if( boxLimite <= otherParentInBox.getZ() ){
                                otherParentInBox.setZ( otherParentInBox.getZ() - boxLimite );
                            }


                            const MortonIndex mortonOtherParent = otherParentInBox.getMortonIndex() << 3;
                            // Get child
                            CellClass** const cells = getCellPt(mortonOtherParent, inLevel);

                            // If there is one or more child
                            if(cells){
                                // For each child
                                for(int idxCousin = 0 ; idxCousin < 8 ; ++idxCousin){
                                    if(cells[idxCousin]){
                                        const int xdiff  = ((otherParent.getX()<<1) | ( (idxCousin>>2) & 1)) - workingCell.getX();
                                        const int ydiff  = ((otherParent.getY()<<1) | ( (idxCousin>>1) & 1)) - workingCell.getY();
                                        const int zdiff  = ((otherParent.getZ()<<1) | (idxCousin&1))         - workingCell.getZ();

                                        // Test if it is a direct neighbor
                                        if(FMath::Abs(xdiff) > neighSeparation || FMath::Abs(ydiff) > neighSeparation || FMath::Abs(zdiff) > neighSeparation){
                                            // add to neighbors
                                            inNeighbors[idxNeighbors] = cells[idxCousin];
                                            inNeighborPositions[idxNeighbors] = (((xdiff+3) * 7) + (ydiff+3)) * 7 + zdiff + 3;
                                            ++idxNeighbors;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return idxNeighbors;
        }
    }


    /** This function return a cell (if it exists) from a morton index and a level
     * @param inIndex the index of the desired cell
     * @param inLevel the level of the desired cell (cannot be inferred from the index)
     * @return the cell if it exist or null (0)
     *
     */
    ContainerClass* getLeafSrc(const MortonIndex inIndex){
        SubOctreeTypes workingTree;
        workingTree.tree = this->root;
        const MortonIndex treeSubLeafMask = ~(~0x00LL << (3 *  workingTree.tree->getSubOctreeHeight() ));

        // Find the suboctree a the correct level
        while(leafIndex >= workingTree.tree->getSubOctreeHeight() + workingTree.tree->getSubOctreePosition()) {
            // compute the leaf index
            const MortonIndex fullIndex = inIndex >> (3 * (leafIndex + 1  - (workingTree.tree->getSubOctreeHeight() + workingTree.tree->getSubOctreePosition()) ) );
            // point to next suboctree
            workingTree.tree = workingTree.middleTree->leafs(int(treeSubLeafMask & fullIndex));
            if(!workingTree.tree) return nullptr;
        }

        // compute correct index in the array
        const MortonIndex treeLeafMask = ~(~0x00LL << (3 *  (leafIndex + 1 - workingTree.tree->getSubOctreePosition()) ));
        return workingTree.leafTree->getLeafSrc(int(treeLeafMask & inIndex));
    }

    /** This function fill an array with the neighbors of a cell
     * @param inNeighbors the array to store the elements
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getLeafsNeighbors(ContainerClass* inNeighbors[27], const FTreeCoordinate& center, const int inLevel){
        memset( inNeighbors, 0 , 27 * sizeof(ContainerClass*));
        const int boxLimite = FMath::pow2(inLevel);

        int idxNeighbors = 0;

        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(center.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(center.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(center.getZ() + idxZ,0,boxLimite)) continue;

                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){
                        const FTreeCoordinate other(center.getX() + idxX,center.getY() + idxY,center.getZ() + idxZ);
                        const MortonIndex mortonOther = other.getMortonIndex();
                        // get cell
                        ContainerClass* const leaf = getLeafSrc(mortonOther);
                        // add to list if not null
                        if(leaf){
                            inNeighbors[(((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1] = leaf;
                            ++idxNeighbors;
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }

    /** This function fill an array with the neighbors of a cell
     * @param inNeighbors the array to store the elements
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getLeafsNeighbors(ContainerClass* inNeighbors[26], int inNeighborPositions[26], const FTreeCoordinate& center, const int inLevel){
        const int boxLimite = FMath::pow2(inLevel);

        int idxNeighbors = 0;

        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(center.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(center.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(center.getZ() + idxZ,0,boxLimite)) continue;

                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){
                        const FTreeCoordinate other(center.getX() + idxX,center.getY() + idxY,center.getZ() + idxZ);
                        const MortonIndex mortonOther = other.getMortonIndex();
                        // get cell
                        ContainerClass* const leaf = getLeafSrc(mortonOther);
                        // add to list if not null
                        if(leaf){
                            inNeighbors[idxNeighbors] = leaf;
                            inNeighborPositions[idxNeighbors] = (((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1;
                            ++idxNeighbors;
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }


    /** This function fill an array with the neighbors of a cell
     * @param inNeighbors the array to store the elements
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getLeafsNeighbors(const CellClass*  inNeighbors[27], const FTreeCoordinate& center, const int inLevel){
        memset( inNeighbors, 0 , 27 * sizeof(CellClass*));
        const int boxLimite = FMath::pow2(inLevel);

        int idxNeighbors = 0;

        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(center.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(center.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(center.getZ() + idxZ,0,boxLimite)) continue;

                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){
                        const FTreeCoordinate other(center.getX() + idxX,center.getY() + idxY,center.getZ() + idxZ);
                        const MortonIndex mortonOther = other.getMortonIndex();
                        // get cell
                        CellClass** const leaf = getCellPt(mortonOther, inLevel);

                        // add to list if not null
                        if(leaf){
                            inNeighbors[(((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1] = leaf;
                            ++idxNeighbors;
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }

    /** This function fill an array with the neighbors of a cell
     * @param inNeighbors the array to store the elements
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getLeafsNeighbors(const CellClass*  inNeighbors[26], int inNeighborPositions[26], const FTreeCoordinate& center, const int inLevel){
        const int boxLimite = FMath::pow2(inLevel);

        int idxNeighbors = 0;

        // We test all cells around
        for(int idxX = -1 ; idxX <= 1 ; ++idxX){
            if(!FMath::Between(center.getX() + idxX,0,boxLimite)) continue;

            for(int idxY = -1 ; idxY <= 1 ; ++idxY){
                if(!FMath::Between(center.getY() + idxY,0,boxLimite)) continue;

                for(int idxZ = -1 ; idxZ <= 1 ; ++idxZ){
                    if(!FMath::Between(center.getZ() + idxZ,0,boxLimite)) continue;

                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){
                        const FTreeCoordinate other(center.getX() + idxX,center.getY() + idxY,center.getZ() + idxZ);
                        const MortonIndex mortonOther = other.getMortonIndex();
                        // get cell
                        CellClass** const leaf = getCellPt(mortonOther, inLevel);

                        // add to list if not null
                        if(leaf){
                            inNeighbors[idxNeighbors] = leaf;
                            inNeighborPositions[idxNeighbors] = (((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1;
                            ++idxNeighbors;
                        }
                    }
                }
            }
        }

        return idxNeighbors;
    }


    /** This function fill an array with the neighbors of a cell
     * @param inNeighbors the array to store the elements
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getPeriodicLeafsNeighbors(ContainerClass* inNeighbors[27], FTreeCoordinate outOffsets[27], bool*const isPeriodic,
                const FTreeCoordinate& center, const int inLevel, const int inDirection){

        const int boxLimite = FMath::pow2(inLevel);

        if( center.getX() != 0 && center.getY() != 0 && center.getZ() != 0 &&
                center.getX() != boxLimite - 1 && center.getY() != boxLimite - 1 && center.getZ() != boxLimite - 1 ){
            (*isPeriodic) = false;
            return getLeafsNeighbors(inNeighbors, center, inLevel);
        }

        (*isPeriodic) = true;
        memset(inNeighbors , 0 , 27 * sizeof(ContainerClass*));
        int idxNeighbors = 0;

        const int startX = (TestPeriodicCondition(inDirection, DirMinusX) || center.getX() != 0 ?-1:0);
        const int endX = (TestPeriodicCondition(inDirection, DirPlusX) || center.getX() != boxLimite - 1 ?1:0);
        const int startY = (TestPeriodicCondition(inDirection, DirMinusY) || center.getY() != 0 ?-1:0);
        const int endY = (TestPeriodicCondition(inDirection, DirPlusY) || center.getY() != boxLimite - 1 ?1:0);
        const int startZ = (TestPeriodicCondition(inDirection, DirMinusZ) || center.getZ() != 0 ?-1:0);
        const int endZ = (TestPeriodicCondition(inDirection, DirPlusZ) || center.getZ() != boxLimite - 1 ?1:0);
        int otherX,otherY,otherZ;
        FTreeCoordinate other;

        int xoffset = 0, yoffset = 0, zoffset = 0;
        // We test all cells around
        for(int idxX = startX ; idxX <= endX ; ++idxX){
            otherX = center.getX() + idxX ; xoffset = 0 ;
            if( otherX < 0 ){
                otherX += boxLimite ;
                xoffset = -1;
            }
            else if( boxLimite <= otherX ){
                otherX -= boxLimite ;
                xoffset = 1;
            }
            other.setX(otherX);
            for(int idxY = startY ; idxY <= endY ; ++idxY){
                otherY = center.getY() + idxY ;
                yoffset = 0 ;
                if( otherY < 0 ){
                    otherY += boxLimite ;
                    yoffset = -1;
                }
                else if( boxLimite <= otherY ){
                    otherY -= boxLimite ;
                    yoffset = 1;
                }
                other.setY(otherY);
                for(int idxZ = startZ ; idxZ <= endZ ; ++idxZ){
                    zoffset = 0 ;
                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){ //  !( idxX !=0  && idxY != 0  &&idxZ != 0  )
                        otherZ = center.getZ() + idxZ ;

                        if( otherZ < 0 ){
                            otherZ += boxLimite ;
                            zoffset = -1;
                        }
                        else if( boxLimite <= otherZ ){
                            otherZ -= boxLimite ;
                            zoffset = 1;
                        }
                        other.setZ(otherZ);

                        const MortonIndex mortonOther = other.getMortonIndex();
                        // get cell
                        ContainerClass* const leaf = getLeafSrc(mortonOther);
                        // add to list if not null
                        if(leaf){
                            const int index = (((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1;
                            inNeighbors[index] = leaf;
                            outOffsets[index].setPosition(xoffset,yoffset,zoffset);

                            ++idxNeighbors;
                        }  // if(leaf)
                    } // if( idxX || idxY || idxZ )
                }
            }
        }

        return idxNeighbors;
    }

    /** This function fill an array with the neighbors of a cell
     * @param inNeighbors the array to store the elements
     * @param inIndex the index of the element we want the neighbors
     * @param inLevel the level of the element
     * @return the number of neighbors
     */
    int getPeriodicLeafsNeighbors(ContainerClass* inNeighbors[26], int inNeighborPositions[26], FTreeCoordinate outOffsets[26], bool*const isPeriodic,
    const FTreeCoordinate& center, const int inLevel, const int inDirection){

        const int boxLimite = FMath::pow2(inLevel);

        if( center.getX() != 0 && center.getY() != 0 && center.getZ() != 0 &&
                center.getX() != boxLimite - 1 && center.getY() != boxLimite - 1 && center.getZ() != boxLimite - 1 ){
            (*isPeriodic) = false;
            return getLeafsNeighbors(inNeighbors, inNeighborPositions, center, inLevel);
        }

        (*isPeriodic) = true;
        int idxNeighbors = 0;

        const int startX = (TestPeriodicCondition(inDirection, DirMinusX) || center.getX() != 0 ?-1:0);
        const int endX = (TestPeriodicCondition(inDirection, DirPlusX) || center.getX() != boxLimite - 1 ?1:0);
        const int startY = (TestPeriodicCondition(inDirection, DirMinusY) || center.getY() != 0 ?-1:0);
        const int endY = (TestPeriodicCondition(inDirection, DirPlusY) || center.getY() != boxLimite - 1 ?1:0);
        const int startZ = (TestPeriodicCondition(inDirection, DirMinusZ) || center.getZ() != 0 ?-1:0);
        const int endZ = (TestPeriodicCondition(inDirection, DirPlusZ) || center.getZ() != boxLimite - 1 ?1:0);
        int otherX,otherY,otherZ;
        FTreeCoordinate other;

        int xoffset = 0, yoffset = 0, zoffset = 0;
        // We test all cells around
        for(int idxX = startX ; idxX <= endX ; ++idxX){
            otherX = center.getX() + idxX ; xoffset = 0 ;
            if( otherX < 0 ){
                otherX += boxLimite ;
                xoffset = -1;
            }
            else if( boxLimite <= otherX ){
                otherX -= boxLimite ;
                xoffset = 1;
            }
            other.setX(otherX);
            for(int idxY = startY ; idxY <= endY ; ++idxY){
                otherY = center.getY() + idxY ;
                yoffset = 0 ;
                if( otherY < 0 ){
                    otherY += boxLimite ;
                    yoffset = -1;
                }
                else if( boxLimite <= otherY ){
                    otherY -= boxLimite ;
                    yoffset = 1;
                }
                other.setY(otherY);
                for(int idxZ = startZ ; idxZ <= endZ ; ++idxZ){
                    zoffset = 0 ;
                    // if we are not on the current cell
                    if( idxX || idxY || idxZ ){ //  !( idxX !=0  && idxY != 0  &&idxZ != 0  )
                        otherZ = center.getZ() + idxZ ;

                        if( otherZ < 0 ){
                            otherZ += boxLimite ;
                            zoffset = -1;
                        }
                        else if( boxLimite <= otherZ ){
                            otherZ -= boxLimite ;
                            zoffset = 1;
                        }
                        other.setZ(otherZ);

                        const MortonIndex mortonOther = other.getMortonIndex();
                        // get cell
                        ContainerClass* const leaf = getLeafSrc(mortonOther);
                        // add to list if not null
                        if(leaf){
                            inNeighbors[idxNeighbors] = leaf;
                            outOffsets[idxNeighbors].setPosition(xoffset,yoffset,zoffset);
                            inNeighborPositions[idxNeighbors] = (((idxX + 1) * 3) + (idxY +1)) * 3 + idxZ + 1;

                            ++idxNeighbors;
                        }  // if(leaf)
                    } // if( idxX || idxY || idxZ )
                }
            }
        }

        return idxNeighbors;
    }


    /////////////////////////////////////////////////////////
    // Lambda function to apply to all member
    /////////////////////////////////////////////////////////

    /**
     * @brief forEachLeaf iterate on the leaf and apply the function
     * @param function
     */
    void forEachLeaf(std::function<void(LeafClass*)> function){
        if(isEmpty()){
            return;
        }

        Iterator octreeIterator(this);
        octreeIterator.gotoBottomLeft();

        do{
            function(octreeIterator.getCurrentLeaf());
        } while(octreeIterator.moveRight());
    }

    /**
     * @brief forEachLeaf iterate on the cell and apply the function
     * @param function
     */
    void forEachCell(std::function<void(CellClass*)> function){
        if(isEmpty()){
            return;
        }

        Iterator octreeIterator(this);
        octreeIterator.gotoBottomLeft();

        Iterator avoidGoLeft(octreeIterator);

        for(int idx = this->height-1 ; idx >= 1 ; --idx ){
            do{
                function(octreeIterator.getCurrentCell());
            } while(octreeIterator.moveRight());
            avoidGoLeft.moveUp();
            octreeIterator = avoidGoLeft;
        }
    }

    /**
     * @brief forEachLeaf iterate on the cell and apply the function
     * @param function
     */
    void forEachCellWithLevel(std::function<void(CellClass*,const int)> function){
        if(isEmpty()){
            return;
        }

        Iterator octreeIterator(this);
        octreeIterator.gotoBottomLeft();

        Iterator avoidGoLeft(octreeIterator);

        for(int idx = this->height-1 ; idx >= 1 ; --idx ){
            do{
                function(octreeIterator.getCurrentCell(),idx);
            } while(octreeIterator.moveRight());
            avoidGoLeft.moveUp();
            octreeIterator = avoidGoLeft;
        }
    }

    /**
     * @brief forEachLeaf iterate on the cell and apply the function
     * @param function
     */
    void forEachCellLeaf(std::function<void(CellClass*,LeafClass*)> function){
        if(isEmpty()){
            return;
        }

        Iterator octreeIterator(this);
        octreeIterator.gotoBottomLeft();

        do{
            function(octreeIterator.getCurrentCell(),octreeIterator.getCurrentLeaf());
        } while(octreeIterator.moveRight());
    }
};

#endif //FOCTREE_HPP
