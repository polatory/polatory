// See LICENCE file at project root
// Keep in private GIT

#ifndef FUNIFTENSOR_HPP
#define FUNIFTENSOR_HPP

#include "../../Utils/FMath.hpp"

#include "FUnifRoots.hpp"
#include "../Interpolation/FInterpTensor.hpp"


/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * Please read the license
 */



/**
 * @class FUnifTensor
 *
 * The class FUnifTensor provides function considering the tensor product
 * interpolation.
 *
 * @tparam ORDER interpolation order \f$\ell\f$
 */
template <class FReal, int ORDER>
class FUnifTensor : public FInterpTensor<FReal, ORDER,FUnifRoots<FReal, ORDER>>
{
  enum {nnodes = TensorTraits<ORDER>::nnodes,
        rc = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1)};
  typedef FUnifRoots<FReal, ORDER> BasisType;
  typedef FInterpTensor<FReal, ORDER,BasisType> ParentTensor;

 public:

  /**
   * Sets the diff of ids of the coordinates of all \f$\ell^6\f$ interpolation
   * nodes duet
   *
   * @param[out] NodeIdsDiff diff of ids of coordinates of interpolation nodes
   */
  static
    void setNodeIdsDiff(unsigned int NodeIdsDiff[nnodes*nnodes])
  {
    unsigned int node_ids[nnodes][3];
    ParentTensor::setNodeIds(node_ids);

    for (unsigned int i=0; i<nnodes; ++i) {
      for (unsigned int j=0; j<nnodes; ++j) {
        // 0 <= id < 2*ORDER-1
        unsigned int idl = node_ids[i][0]-node_ids[j][0]+ORDER-1;
        unsigned int idm = node_ids[i][1]-node_ids[j][1]+ORDER-1;
        unsigned int idn = node_ids[i][2]-node_ids[j][2]+ORDER-1;
        NodeIdsDiff[i*nnodes+j]
          = idn*(2*ORDER-1)*(2*ORDER-1) + idm*(2*ORDER-1) + idl;
      }
    }
  }

   /**
   * Compute roots index pairs for each of the (2\ell-1)^3 pairs 
   *
   * @param[out] NodeIdsPairs diff of ids of coordinates of interpolation nodes
   */ 
  static
    void setNodeIdsPairs(unsigned int NodeIdsPairs[rc][2])
  {

    unsigned int li,lj,mi,mj,ni,nj;
    unsigned int ido=0;

    // If the diff i-j between indices is <0 (i.e. positive counter < ORDER-1)
    // then source index is ORDER-1 
    // else it is the first
    // we deduce the target index by simply considering i-j=counter-(ORDER-1) 
    for(unsigned int l=0; l<2*ORDER-1; ++l){

      // l=0:(2*order-1) => li-lj=-(order-1):(order-1)
      // Convention:
      // lj=order-1 & li=0:order-1 => li-lj=1-order:0
      // lj=1 & li=0:order-1 => li-lj=1:order-1
      if(l<ORDER-1) lj=ORDER-1; else lj=0;
      li=(l-(ORDER-1))+lj;

      for(unsigned int m=0; m<2*ORDER-1; ++m){

        if(m<ORDER-1) mj=ORDER-1; else mj=0;
        mi=(m-(ORDER-1))+mj;

        for(unsigned int n=0; n<2*ORDER-1; ++n){

          if(n<ORDER-1) nj=ORDER-1; else nj=0;
          ni=(n-(ORDER-1))+nj;

          NodeIdsPairs[ido][0]=li*ORDER*ORDER + mi*ORDER + ni;
          NodeIdsPairs[ido][1]=lj*ORDER*ORDER + mj*ORDER + nj;

          ido++;
        }
      }
    }

  }

   /**
   *  Compute permutation vector used to reorder first row of circulant matrix
   *
   * @param[out] perm
   */ 
  static
    void setStoragePermutation(unsigned int perm[rc])
  {
    for(unsigned int p=0; p<rc; ++p){
      // permutation to order WHILE computing the entire 1st row
      if(p<rc-1) perm[p]=p+1;
      else perm[p]=p+1-rc;
      //    std::cout << "perm["<< p << "]="<< perm[p] << std::endl;
    }

  }

};


#endif /*FUNIFTENSOR_HPP*/
