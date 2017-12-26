// See LICENCE file at project root

#ifndef FABSTRACTBALANCEALGORITHM_H
#define FABSTRACTBALANCEALGORITHM_H


/**
 * @author Cyrille Piacibello
 * @class FAbstractBalanceAlgorithm
 *
 * @brief This class provide the methods that are used to balance a
 * tree FMpiTreeBuilder::EqualizeAndFillTree
 */
class FAbstractBalanceAlgorithm{
public:
  virtual ~FAbstractBalanceAlgorithm(){
  }

  /**
   * @brief Give the right leaves (ie the min) of the interval that
   * will be handle by idxOfProc
   * @param numberOfLeaves Total number of leaves that exist.
   * @param numberOfPartPerLeaf Array of lenght numberOfLeaves containing the number of particles in each leaf
   * @param numberOfPart Number of particles in the whole field
   * @param idxOfLeaves Array of lenght numberOfLeaves containing the Morton Index of each Leaf
   * @param numberOfProc Number of MPI processus that will handle the Octree
   * @param idxOfProc Idx of the proc calling.
   */
  virtual FSize getRight(const FSize numberOfLeaves,
                         const int numberOfProc, const int idxOfProc) = 0;

  /**
   * @brief Give the Leaft leaves (ie the max) of the interval that
   * will be handle by idxOfProc
   * @param numberOfLeaves Total number of leaves that exist.
   * @param numberOfPartPerLeaf Array of lenght numberOfLeaves containing the number of particles in each leaf
   * @param numberOfPart Number of particles in the whole field
   * @param idxOfLeaves Array of lenght numberOfLeaves containing the Morton Index of each Leaf
   * @param numberOfProc Number of MPI processus that will handle the Octree
   * @param idxOfProc Idx of the proc calling.
   */
  virtual FSize getLeft(const FSize numberOfLeaves,
                        const int numberOfProc, const int idxOfProc) = 0;

};

#endif //FABSTRACTBALANCEALGORITHM_H
