// See LICENCE file at project root

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
