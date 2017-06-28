#ifndef FPARFOREACHOCTREE_HPP
#define FPARFOREACHOCTREE_HPP

#include "FOctree.hpp"

#include <functional>

namespace FParForEachOctree {

/////////////////////////////////////////////////////////
// Lambda function to apply to all member
/////////////////////////////////////////////////////////

/**
 * @brief forEachLeaf iterate on the leaf and apply the function
 * @param function
 */
template< template<class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass > class FOctree,
          class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass,
          class FunctionTemplate>
void forEachLeaf(FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>* tree, FunctionTemplate function){
    #pragma omp parallel
    {
        #pragma omp single
        {
            typename FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();

            do{
                LeafClass* lf = octreeIterator.getCurrentLeaf();
                #pragma omp task firstprivate(lf)
                {
                    function(lf);
                }
            } while(octreeIterator.moveRight());

            #pragma omp taskwait
        }
    }
}

/**
 * @brief forEachLeaf iterate on the cell and apply the function
 * @param function
 */
template<  template<class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass > class FOctree,
          class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass,
          class FunctionTemplate>
void forEachCell(FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>* tree, FunctionTemplate function){
    #pragma omp parallel
    {
        #pragma omp single
        {
            typename FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();

            typename FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>::Iterator avoidGoLeft(octreeIterator);

            for(int idx = tree->getHeight()-1 ; idx >= 1 ; --idx ){
                do{
                    CellClass* cell = octreeIterator.getCurrentCell();
                    #pragma omp task firstprivate(cell)
                    {
                        function(cell);
                    }
                } while(octreeIterator.moveRight());
                avoidGoLeft.moveUp();
                octreeIterator = avoidGoLeft;
            }

            #pragma omp taskwait
        }
    }
}

/**
 * @brief forEachLeaf iterate on the cell and apply the function
 * @param function
 */
template<  template<class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass > class FOctree,
          class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass,
          class FunctionTemplate>
void forEachCellWithLevel(FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>* tree, FunctionTemplate function){
    #pragma omp parallel
    {
        #pragma omp single
        {
            typename FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();

            typename FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>::Iterator avoidGoLeft(octreeIterator);

            for(int idx = tree->getHeight()-1 ; idx >= 1 ; --idx ){
                do{
                    CellClass* cell = octreeIterator.getCurrentCell();
                    #pragma omp task firstprivate(cell, idx)
                    {
                        function(cell,idx);
                    }
                } while(octreeIterator.moveRight());
                avoidGoLeft.moveUp();
                octreeIterator = avoidGoLeft;
            }

            #pragma omp taskwait
        }
    }
}

/**
 * @brief forEachLeaf iterate on the cell and apply the function
 * @param function
 */
template<  template<class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass > class FOctree,
          class FReal, class CellClass, class ContainerClass, class LeafClass, class CellAllocatorClass,
          class FunctionTemplate>
void forEachCellLeaf(FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>* tree, FunctionTemplate function){
    #pragma omp parallel
    {
        #pragma omp single
        {
            typename FOctree<FReal, CellClass,ContainerClass,LeafClass,CellAllocatorClass>::Iterator octreeIterator(tree);
            octreeIterator.gotoBottomLeft();

            do{
                CellClass* cell = octreeIterator.getCurrentCell();
                LeafClass* lf = octreeIterator.getCurrentLeaf();
                #pragma omp task firstprivate(cell, lf)
                {
                    function(cell,lf);
                }
            } while(octreeIterator.moveRight());

            #pragma omp taskwait
        }
    }
}

}

#endif // FPARFOREACHOCTREE_HPP
