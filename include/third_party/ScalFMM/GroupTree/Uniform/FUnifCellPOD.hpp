#ifndef FUNIFCELLPOD_HPP
#define FUNIFCELLPOD_HPP

#include "../../Utils/FGlobal.hpp"
#include "../Core/FBasicCellPOD.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"
#include "../../Kernels/Uniform//FUnifTensor.hpp"
#include "../../Utils/FComplex.hpp"

typedef FBasicCellPOD FUnifCellPODCore;

template <class FReal, int ORDER, int NRHS = 1, int NLHS = 1, int NVALS = 1>
struct alignas(FStarPUDefaultAlign::StructAlign) FUnifCellPODPole {
    static const int VectorSize = TensorTraits<ORDER>::nnodes;
    static const int TransformedVectorSize = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1);
    FReal multipole_exp[NRHS * NVALS * VectorSize]; //< Multipole expansion
    FComplex<FReal> transformed_multipole_exp[NRHS * NVALS * TransformedVectorSize];
};

template <class FReal, int ORDER, int NRHS = 1, int NLHS = 1, int NVALS = 1>
struct alignas(FStarPUDefaultAlign::StructAlign) FUnifCellPODLocal {
    static const int VectorSize = TensorTraits<ORDER>::nnodes;
    static const int TransformedVectorSize = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1);
    FComplex<FReal>     transformed_local_exp[NLHS * NVALS * TransformedVectorSize];
    FReal     local_exp[NLHS * NVALS * VectorSize]; //< Local expansion
};

template <class FReal, int ORDER, int NRHS = 1, int NLHS = 1, int NVALS = 1>
class FUnifCellPOD {
  FUnifCellPODCore* symb;
  FUnifCellPODPole<FReal,ORDER,NRHS,NLHS,NVALS>* up;
  FUnifCellPODLocal<FReal,ORDER,NRHS,NLHS,NVALS>* down;

public:
  FUnifCellPOD(FUnifCellPODCore* inSymb, FUnifCellPODPole<FReal,ORDER,NRHS,NLHS,NVALS>* inUp,
            FUnifCellPODLocal<FReal,ORDER,NRHS,NLHS,NVALS>* inDown): symb(inSymb), up(inUp), down(inDown){
  }

  FUnifCellPOD()
      : symb(nullptr), up(nullptr), down(nullptr){
  }


  /** To get the morton index */
  MortonIndex getMortonIndex() const {
      return symb->mortonIndex;
  }

  /** To set the morton index */
  void setMortonIndex(const MortonIndex inMortonIndex) {
      symb->mortonIndex = inMortonIndex;
  }

  /** To get the cell level */
  int getLevel() const {
      return symb->level;
  }

  /** To set the cell level */
  void setLevel(const int level) {
      symb->level = level;
  }

  /** To get the position */
  FTreeCoordinate getCoordinate() const {
      return FTreeCoordinate(symb->coordinates[0],
              symb->coordinates[1], symb->coordinates[2]);
  }

  /** To set the position */
  void setCoordinate(const FTreeCoordinate& inCoordinate) {
      symb->coordinates[0] = inCoordinate.getX();
      symb->coordinates[1] = inCoordinate.getY();
      symb->coordinates[2] = inCoordinate.getZ();
  }

  /** To set the position from 3 FReals */
  void setCoordinate(const int inX, const int inY, const int inZ) {
      symb->coordinates[0] = inX;
      symb->coordinates[1] = inY;
      symb->coordinates[2] = inZ;
  }

  /** Get Multipole */
  const FReal* getMultipole(const int inRhs) const
  {	return up->multipole_exp + inRhs*up->VectorSize;
  }
  /** Get Local */
  const FReal* getLocal(const int inRhs) const{
    return down->local_exp + inRhs*down->VectorSize;
  }

  /** Get Multipole */
  FReal* getMultipole(const int inRhs){
    return up->multipole_exp + inRhs*up->VectorSize;
  }
  /** Get Local */
  FReal* getLocal(const int inRhs){
    return down->local_exp + inRhs*down->VectorSize;
  }

  /** To get the leading dim of a vec */
  int getVectorSize() const{
    return down->VectorSize;
  }

  /** Get Transformed Multipole */
  const FComplex<FReal>* getTransformedMultipole(const int inRhs) const{
    return up->transformed_multipole_exp + inRhs*up->TransformedVectorSize;
  }
  /** Get Transformed Local */
  const FComplex<FReal>* getTransformedLocal(const int inRhs) const{
    return down->transformed_local_exp + inRhs*down->TransformedVectorSize;
  }

  /** Get Transformed Multipole */
  FComplex<FReal>* getTransformedMultipole(const int inRhs){
    return up->transformed_multipole_exp + inRhs*up->TransformedVectorSize;
  }
  /** Get Transformed Local */
  FComplex<FReal>* getTransformedLocal(const int inRhs){
    return down->transformed_local_exp + inRhs*down->TransformedVectorSize;
  }

  /** To get the leading dim of a vec */
  int getTransformedVectorSize() const{
    return down->TransformedVectorSize;
  }

  /** Make it like the begining */
  void resetToInitialState(){
    memset(up->multipole_exp, 0, sizeof(FReal) * NRHS * NVALS * up->VectorSize);
    memset(down->local_exp, 0, sizeof(FReal) * NLHS * NVALS * down->VectorSize);
    memset(up->transformed_multipole_exp, 0,
           sizeof(FComplex<FReal>) * NRHS * NVALS * up->TransformedVectorSize);
    memset(down->transformed_local_exp, 0,
           sizeof(FComplex<FReal>) * NLHS * NVALS * down->TransformedVectorSize);
  }
};

#endif // FUNIFCELLPOD_HPP
