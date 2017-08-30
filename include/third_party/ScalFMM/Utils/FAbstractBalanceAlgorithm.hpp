// ===================================================================================
// Copyright ScalFmm 2016 INRIA, Olivier Coulaud, Bérenger Bramas,
// Matthias Messner olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the
// FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
// An extension to the license is given to allow static linking of scalfmm
// inside a proprietary application (no matter its license).
// See the main license file for more details.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================

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
