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

#ifndef FLEAFBALANCE_H
#define FLEAFBALANCE_H

#include "./FAbstractBalanceAlgorithm.hpp"
#include "../Utils/FMath.hpp"

/**
 * @author Cyrille Piacibello
 * @class FLeafBalance
 *
 * @brief This class inherits from FAbstractBalanceAlgorithm. It
 * provides balancing methods based on leaf numbers only.
 */
class FLeafBalance : public FAbstractBalanceAlgorithm{

public:
  /**
   * Does not need the number of particles. Just divide the leaves
   * over processus
   */
  FSize getRight(const FSize numberOfLeaves,
                 const int numberOfProc, const int idxOfProc){
    const double step = (double(numberOfLeaves) / double(numberOfProc));
    const FSize res = FSize(FMath::Ceil(step * double(idxOfProc+1)));
    if(res > numberOfLeaves) return numberOfLeaves;
    else return res;
  }

  /**
   * Does not need the number of particles. Just divide the leaves
   * over processus
   */
  FSize getLeft(const FSize numberOfLeaves,
                const int numberOfProc, const int idxOfProc){
    const double step = (double(numberOfLeaves) / double(numberOfProc));
    return FSize(FMath::Ceil(step * double(idxOfProc)));
  }

};


#endif // FLEAFBALANCE_H
