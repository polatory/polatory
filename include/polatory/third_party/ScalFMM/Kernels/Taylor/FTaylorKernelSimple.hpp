// See LICENCE file at project root
#ifndef FTAYLORKERNEL_HPP
#define FTAYLORKERNEL_HPP

#include "../../Components/FAbstractKernels.hpp"
#include "../../Utils/FMemUtils.hpp"
#include "../../Utils/FLog.hpp"

#include "../P2P/FP2PR.hpp"

/**
 * @author Cyrille Piacibello
 * @class FTaylorKernel
 *
 * @brief This kernel is an implementation of the different operators
 * needed to compute the Fast Multipole Method using Taylor Expansion
 * for the Far fields interaction.
 */


//TODO spécifier les arguments.
template< class FReal, class CellClass, class ContainerClass, int P, int order>
class FTaylorKernel : public FAbstractKernels<CellClass,ContainerClass> {

private:
    //Size of the multipole and local vectors
    static const int SizeVector = ((P+1)*(P+2)*(P+3))*order/6;


    ////////////////////////////////////////////////////
    // Object Attributes
    ////////////////////////////////////////////////////
    const FReal boxWidth;               //< the box width at leaf level
    const int   treeHeight;             //< The height of the tree
    const FReal widthAtLeafLevel;       //< width of box at leaf level
    const FReal widthAtLeafLevelDiv2;   //< width of box at leaf leve div 2
    const FPoint<FReal> boxCorner;             //< position of the box corner

    FReal factorials[2*P+1];             //< This contains the factorial until P
    FReal arrayDX[P+2],arrayDY[P+2],arrayDZ[P+2] ; //< Working arrays
    static const int  sizeDerivative = (2*P+1)*(P+1)*(2*P+3)/3;
    FReal _PsiVector[sizeDerivative];
    FReal _coeffPoly[SizeVector];

    ////////////////////////////////////////////////////
    // Private method
    ////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////
    // Precomputation
    ///////////////////////////////////////////////////////

    /** Compute the factorial from 0 to P
   * Then the data is accessible in factorials array:
   * factorials[n] = n! with n <= P
   */
    void precomputeFactorials(){
        factorials[0] = 1.0;
        FReal fidx = 1.0;
        for(int idx = 1 ; idx <= 2*P ; ++idx, ++fidx){
            factorials[idx] = fidx * factorials[idx-1];
        }
    }


    /** Return the position of a leaf from its tree coordinate */
    FPoint<FReal> getLeafCenter(const FTreeCoordinate coordinate) const {
        return FPoint<FReal>(
                    FReal(coordinate.getX()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getX(),
                    FReal(coordinate.getY()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getY(),
                    FReal(coordinate.getZ()) * widthAtLeafLevel + widthAtLeafLevelDiv2 + boxCorner.getZ());
    }

    /**
   * @brief Return the position of the center of a cell from its tree
   *  coordinate
   * @param FTreeCoordinate
   * @param inLevel the current level of Cell
   */
    FPoint<FReal> getCellCenter(const FTreeCoordinate coordinate, int inLevel)
    {

        //Set the boxes width needed
        FReal widthAtCurrentLevel = widthAtLeafLevel*FReal(1 << (treeHeight-(inLevel+1)));
        FReal widthAtCurrentLevelDiv2 = widthAtCurrentLevel/FReal(2);

        //Get the coordinate
        int a = coordinate.getX();
        int b = coordinate.getY();
        int c = coordinate.getZ();

        //Set the center real coordinates from box corner and widths.
        FReal X = boxCorner.getX() + FReal(a)*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
        FReal Y = boxCorner.getY() + FReal(b)*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
        FReal Z = boxCorner.getZ() + FReal(c)*widthAtCurrentLevel + widthAtCurrentLevelDiv2;

        FPoint<FReal> cCenter = FPoint<FReal>(X,Y,Z);
        return cCenter;
    }


    /**
   * @brief Incrementation of powers in Taylor expansion
   * Result : ...,[2,0,0],[1,1,0],[1,0,1],[0,2,0]...  3-tuple are sorted
   * by size then alphabetical order.
   */
    void incPowers(int * const FRestrict a, int *const FRestrict b, int *const FRestrict c)
    {
        int t = (*a)+(*b)+(*c);
        if(t==0)
        {a[0]=1;}
        else{ if(t==a[0])
            {a[0]--;  b[0]++;}
            else{ if(t==c[0])
                {a[0]=t+1;  c[0]=0;}
                else{ if(b[0]!=0)
                    {b[0]--; c[0]++;}
                    else{
                        b[0]=c[0]+1;
                        a[0]--;
                        c[0]=0;
                    }
                }
            }
        }
    }

    /**
   * @brief Give the index of array from the corresponding 3-tuple
   * powers.
   */
    int powerToIdx(const int a,const int b,const int c)
    {
        int t,res,p = a+b+c;
        res  = p*(p+1)*(p+2)/6;
        t    = p - a;
        res += t*(t+1)/2+c;
        return res;
    }

    /* Return the factorial of a number
   */
    FReal fact(const int a){
        if(a<0) {
            printf("fact :: Error factorial negative!! a=%d\n",a);
            return FReal(0);
        }
        FReal result = 1;
        for(int i = 1 ; i <= a ; ++i){
            result *= FReal(i);
        }
        return result;
    }

    /* Return the product of factorial of 3 numbers
   */
    FReal fact3int(const int a,const int b,const int c)
    {
        return ( factorials[a]*factorials[b]* factorials[c]) ;
    }

    /* Return the combine of a paire of number
   * \f[ C^{b}_{a} \f]
   */
    FReal combin(const int& a, const int& b){
        if(a<b)  {printf("combin :: Error combin negative!! a=%d b=%d\n",a,b); exit(-1) ;  }
        return  factorials[a]/  (factorials[b]* factorials[a-b]) ;
    }


    /** @brief Init the derivative array for using of following formula
   * from "A cartesian tree-code for screened coulomb interactions"
   *
   *  @todo METTRE les fonctions pour intialiser la recurrence. \f$x_i\f$ ?? \f$x_i\f$ ??
   *  @todo LA formule ci-dessous n'utilise pas k!
   */

    void initDerivative(const FReal & dx ,const FReal & dy ,const FReal & dz  ,   FReal * tab)
    {
        FReal R2 = dx*dx+dy*dy+dz*dz;
        tab[0]=FReal(1)/FMath::Sqrt(R2);
        FReal R3 = tab[0]/(R2);
        tab[1]= -dx*R3;                 //Derivative in (1,0,0)
        tab[2]= -dy*R3;                 //Derivative in (0,1,0)
        tab[3]= -dz*R3;                 //Derivative in (0,0,1)
        FReal R5 = R3/R2;
        tab[4] = FReal(3)*dx*dx*R5-R3;  //Derivative in (2,0,0)
        tab[5] = FReal(3)*dx*dy*R5;     //Derivative in (1,1,0)
        tab[6] = FReal(3)*dx*dz*R5;     //Derivative in (1,0,1)
        tab[7] = FReal(3)*dy*dy*R5-R3;  //Derivative in (0,2,0)
        tab[8] = FReal(3)*dy*dz*R5;     //Derivative in (0,1,1)
        tab[9] = FReal(3)*dz*dz*R5-R3;  //Derivative in (0,0,2)
    }

    /** @brief Compute and store the derivative for a given tuple.
   *  Derivative are used for the M2L
   *
   *\f[
   * \Psi_{\mathbf{k}}^{c} \left [\left |\mathbf{k}\right |\times \left |
   * \mathbf{x}_i-\mathbf{x}_c\right |^2  \right ]\
   * = (2\times \left |{\mathbf{k}}\right |-1)
   * \sum_{j=0}^{3}\left [ k_j (x_{i_j}-x_{c_j})
   * \Psi_{\mathbf{k}-e_j,i}^{c}\right ]\
   * -(\left |\mathbf{k}\right |-1)   \sum_{j=0}^{3}\left
   * [ k_j(k_j-1) \Psi_{\mathbf{k}-2 e_j,i}^{c} \right]
   * \f]
   *  where    \f$ \mathbf{k} = (k_1,k_2,k_3) \f$
   */
    void computeFullDerivative( FReal  dx,  FReal  dy,  FReal  dz, // Distance from distant center to local center
                                FReal * yetComputed)
    {

        initDerivative(dx,dy,dz,yetComputed);
        FReal dist2 =  dx*dx+dy*dy+dz*dz;
        int idxTarget;                      //Index of current yetComputed entry
        int idxSrc1, idxSrc2, idxSrc3,      //Indexes of needed yetComputed entries
                idxSrc4, idxSrc5, idxSrc6;
        int a=0,b=0,c=0;                    //Powers of expansions

        for(c=3 ; c<=2*P ; ++c){
            //Computation of derivatives Psi_{0,0,c}
            // |x-y|^2 * Psi_{0,0,c} + (2*c-1) * dz *Psi_{0,0,c-1} + (c-1)^2 * Psi_{0,0,c-2} = 0
            idxTarget = powerToIdx(0,0,c);
            idxSrc1 = powerToIdx(0,0,c-1);
            idxSrc2 = powerToIdx(0,0,c-2);
            yetComputed[idxTarget] = -(FReal(2*c-1)*dz*yetComputed[idxSrc1] + FReal((c-1)*(c-1))*yetComputed[idxSrc2])/dist2;
        }
        b=1;
        for(c=2 ; c<=2*P-1 ; ++c){
            //Computation of derivatives Psi_{0,1,c}
            // |x-y|^2 * Psi_{0,1,c} + (2*c) * dz *Psi_{0,1,c-1} + c*(c-1) * Psi_{0,1,c-2} + dy*Psi_{0,0,c} = 0
            idxTarget = powerToIdx(0,1,c);
            idxSrc1 = powerToIdx(0,1,c-1);
            idxSrc2 = powerToIdx(0,1,c-2);
            idxSrc3 = powerToIdx(0,0,c);
            yetComputed[idxTarget] = -(FReal(2*c)*dz*yetComputed[idxSrc1] + FReal(c*(c-1))*yetComputed[idxSrc2]+ dy*yetComputed[idxSrc3])/dist2;
        }
        b=2;
        for(c=1 ; c<= 2*P-b ; ++c){
            //Computation of derivatives Psi_{0,2,c}
            //|x-y|^2 * Psi_{0,2,c} + (2*c) * dz *Psi_{0,2,c-1} + (c*(c-1)) * Psi_{0,2,c-2} + 3*dy * Psi_{0,1,c} + Psi_{0,0,c}  = 0
            idxTarget = powerToIdx(0,2,c);
            idxSrc1 = powerToIdx(0,2,c-1);
            idxSrc2 = powerToIdx(0,2,c-2);
            idxSrc3 = powerToIdx(0,1,c);
            idxSrc4 = powerToIdx(0,0,c);
            yetComputed[idxTarget] = -(FReal(2*c)*dz*yetComputed[idxSrc1] + FReal(c*(c-1))*yetComputed[idxSrc2]
                                       + FReal(3)*dy*yetComputed[idxSrc3] + yetComputed[idxSrc4])/dist2;
        }
        for(b=3 ; b<= 2*P ; ++b){
            //Computation of derivatives Psi_{0,b,0}
            // |x-y|^2 * Psi_{0,b,0} + (2*b-1) * dy *Psi_{0,b-1,0} + (b-1)^2 * Psi_{0,b-2,c} = 0
            idxTarget = powerToIdx(0,b,0);
            idxSrc1 = powerToIdx(0,b-1,0);
            idxSrc2 = powerToIdx(0,b-2,0);
            yetComputed[idxTarget] = -(FReal(2*b-1)*dy*yetComputed[idxSrc1] + FReal((b-1)*(b-1))*yetComputed[idxSrc2])/dist2;
            for(c=1 ; c<= 2*P-b ; ++c) {
                //Computation of derivatives Psi_{0,b,c}
                //|x-y|^2*Psi_{0,b,c} + (2*c)*dz*Psi_{0,b,c-1} + (c*(c-1))*Psi_{0,b,c-2} + (2*b-1)*dy*Psi_{0,b-1,c} + (b-1)^2 * Psi_{0,b-2,c}  = 0
                idxTarget = powerToIdx(0,b,c);
                idxSrc1 = powerToIdx(0,b,c-1);
                idxSrc2 = powerToIdx(0,b,c-2);
                idxSrc3 = powerToIdx(0,b-1,c);
                idxSrc4 = powerToIdx(0,b-2,c);
                yetComputed[idxTarget] = -(FReal(2*c)*dz*yetComputed[idxSrc1] + FReal(c*(c-1))*yetComputed[idxSrc2]
                                           + FReal(2*b-1)*dy*yetComputed[idxSrc3] + FReal((b-1)*(b-1))*yetComputed[idxSrc4])/dist2;
            }
        }
        a=1;
        b=0;
        for(c=2 ; c<= 2*P-1 ; ++c){
            //Computation of derivatives Psi_{1,0,c}
            //|x-y|^2 * Psi_{1,0,c} + (2*c)*dz*Psi_{1,0,c-1} + c*(c-1)*Psi_{1,0,c-2} + dx*Psi_{0,0,c}
            idxTarget = powerToIdx(1,0,c);
            idxSrc1 = powerToIdx(1,0,c-1);
            idxSrc2 = powerToIdx(1,0,c-2);
            idxSrc3 = powerToIdx(0,0,c);
            yetComputed[idxTarget] = -(FReal(2*c)*dz*yetComputed[idxSrc1] + FReal(c*(c-1))*yetComputed[idxSrc2] + dx*yetComputed[idxSrc3])/dist2;
        }
        b=1;
        //Computation of derivatives Psi_{1,1,1}
        //|x-y|^2 * Psi_{1,1,1} + 2*dz*Psi_{1,1,0} + 2*dy*Psi_{1,0,1} + dx*Psi_{0,1,1}
        idxTarget = powerToIdx(1,1,1);
        idxSrc1 = powerToIdx(1,1,0);
        idxSrc2 = powerToIdx(1,0,1);
        idxSrc3 = powerToIdx(0,1,1);
        yetComputed[idxTarget] = -(FReal(2)*dz*yetComputed[idxSrc1] + FReal(2)*dy*yetComputed[idxSrc2] + dx*yetComputed[idxSrc3])/dist2;
        for(c=2 ; c<= 2*P-2 ; ++c){
            //Computation of derivatives Psi_{1,1,c}
            //|x-y|^2 * Psi_{1,1,c} + (2*c)*dz*Psi_{1,1,c-1} + c*(c-1)*Psi_{1,1,c-2} + 2*dy*Psi_{1,0,c} + dx*Psi_{0,1,c}
            idxTarget = powerToIdx(1,1,c);
            idxSrc1 = powerToIdx(1,1,c-1);
            idxSrc2 = powerToIdx(1,1,c-2);
            idxSrc3 = powerToIdx(1,0,c);
            idxSrc4 = powerToIdx(0,1,c);
            yetComputed[idxTarget] = -(FReal(2*c)*dz*yetComputed[idxSrc1] + FReal(c*(c-1))*yetComputed[idxSrc2]
                                       + FReal(2)*dy*yetComputed[idxSrc3]+ dx*yetComputed[idxSrc4])/dist2;
        }
        for(b=2 ; b<= 2*P-a ; ++b){
            for(c=0 ; c<= 2*P-b-1 ; ++c){
                //Computation of derivatives Psi_{1,b,c}
                //|x-y|^2 * Psi_{1,b,c} + (2*b)*dy*Psi_{1,b-1,c} + b*(b-1)*Psi_{1,b-2,c} + (2*c)*dz*Psi_{1,b,c-1} + c*(c-1)*Psi_{1,b,c-2} + dx*Psi_{0,b,c}
                idxTarget = powerToIdx(1,b,c);
                idxSrc1 = powerToIdx(1,b-1,c);
                idxSrc2 = powerToIdx(1,b-2,c);
                idxSrc3 = powerToIdx(1,b,c-1);
                idxSrc4 = powerToIdx(1,b,c-2);
                idxSrc5 = powerToIdx(0,b,c);
                yetComputed[idxTarget] = -(FReal(2*b)*dy*yetComputed[idxSrc1] + FReal(b*(b-1))*yetComputed[idxSrc2]
                                           + FReal(2*c)*dz*yetComputed[idxSrc3]+ FReal(c*(c-1))*yetComputed[idxSrc4]
                                           + dx*yetComputed[idxSrc5])/dist2;
            }
        }
        for(a=2 ; a<=2*P ; ++a){
            //Computation of derivatives Psi_{a,0,0}
            // |x-y|^2 * Psi_{a,0,0} + (2*a-1) * dx *Psi_{a-1,0,0} + (a-1)^2 * Psi_{a-2,0,0} = 0
            idxTarget = powerToIdx(a,0,0);
            idxSrc1 = powerToIdx(a-1,0,0);
            idxSrc2 = powerToIdx(a-2,0,0);
            yetComputed[idxTarget] = -(FReal(2*a-1)*dx*yetComputed[idxSrc1] + FReal((a-1)*(a-1))*yetComputed[idxSrc2])/dist2;
            if(a <= 2*P-1){
                //Computation of derivatives Psi_{a,0,1}
                // |x-y|^2 * Psi_{a,0,1} + 2*dz*Psi_{a,0,0} + (2*a-1)*dx*Psi_{a-1,0,1} + (a-1)^2*Psi_{a-2,0,1} = 0
                idxSrc1 = idxTarget;
                idxTarget = powerToIdx(a,0,1);
                idxSrc2 = powerToIdx(a-1,0,1);
                idxSrc3 = powerToIdx(a-2,0,1);
                yetComputed[idxTarget] = -(FReal(2)*dz*yetComputed[idxSrc1] + FReal(2*a-1)*dx*yetComputed[idxSrc2] + FReal((a-1)*(a-1))*yetComputed[idxSrc3])/dist2;
                //Computation of derivatives Psi_{a,1,0}
                // |x-y|^2 * Psi_{a,1,0} + 2*dy*Psi_{a,0,0} + (2*a-1)*dx*Psi_{a-1,1,0} + (a-1)^2*Psi_{a-2,1,0} = 0
                idxTarget = powerToIdx(a,1,0);
                idxSrc2 = powerToIdx(a-1,1,0);
                idxSrc3 = powerToIdx(a-2,1,0);
                yetComputed[idxTarget] = -(FReal(2)*dy*yetComputed[idxSrc1] + FReal(2*a-1)*dx*yetComputed[idxSrc2] + FReal((a-1)*(a-1))*yetComputed[idxSrc3])/dist2;
                if(a <= 2*P-2){
                    b=0;
                    for(c=2 ; c <= 2*P-a ; ++c){
                        //Computation of derivatives Psi_{a,0,c}
                        // |x-y|^2 * Psi_{a,0,c} + 2*c*dz*Psi_{a,0,c-1} + c*(c-1)*Psi_{a,0,c-2} + (2*a-1)*dx*Psi_{a-1,0,c} + (a-1)^2*Psi_{a-2,0,c} = 0
                        idxTarget = powerToIdx(a,0,c);
                        idxSrc1 = powerToIdx(a,0,c-1);
                        idxSrc2 = powerToIdx(a,0,c-2);
                        idxSrc3 = powerToIdx(a-1,0,c);
                        idxSrc4 = powerToIdx(a-2,0,c);
                        yetComputed[idxTarget] = -(FReal(2*c)*dz*yetComputed[idxSrc1] + FReal(c*(c-1))*yetComputed[idxSrc2]
                                                   + FReal(2*a-1)*dx*yetComputed[idxSrc3] + FReal((a-1)*(a-1))*yetComputed[idxSrc4])/dist2;
                    }
                    b=1;
                    for(c=1 ; c <= 2*P-a-1 ; ++c){
                        //Computation of derivatives Psi_{a,1,c}
                        // |x-y|^2 * Psi_{a,1,c} + 2*c*dz*Psi_{a,1,c-1} + c*(c-1)*Psi_{a,1,c-2} + 2*a*dx*Psi_{a-1,1,c} + a*(a-1)*Psi_{a-2,1,c} + dy*Psi_{a,0,c}= 0
                        idxTarget = powerToIdx(a,1,c);
                        idxSrc1 = powerToIdx(a,1,c-1);
                        idxSrc2 = powerToIdx(a,1,c-2);
                        idxSrc3 = powerToIdx(a-1,1,c);
                        idxSrc4 = powerToIdx(a-2,1,c);
                        idxSrc5 = powerToIdx(a,0,c);
                        yetComputed[idxTarget] = -(FReal(2*c)*dz*yetComputed[idxSrc1] + FReal(c*(c-1))*yetComputed[idxSrc2]
                                                   + FReal(2*a)*dx*yetComputed[idxSrc3] + FReal(a*(a-1))*yetComputed[idxSrc4]
                                                   + dy*yetComputed[idxSrc5])/dist2;
                    }
                    for(b=2 ; b <= 2*P-a ; ++b){
                        //Computation of derivatives Psi_{a,b,0}
                        // |x-y|^2 * Psi_{a,b,0} + 2*b*dy*Psi_{a,b-1,0} + b*(b-1)*Psi_{a,b-2,0} + (2*a-1)*dx*Psi_{a-1,b,0} + (a-1)^2*Psi_{a-2,b,0} = 0
                        idxTarget = powerToIdx(a,b,0);
                        idxSrc1 = powerToIdx(a,b-1,0);
                        idxSrc2 = powerToIdx(a,b-2,0);
                        idxSrc3 = powerToIdx(a-1,b,0);
                        idxSrc4 = powerToIdx(a-2,b,0);
                        yetComputed[idxTarget] = -(FReal(2*b)*dy*yetComputed[idxSrc1] + FReal(b*(b-1))*yetComputed[idxSrc2]
                                                   + FReal(2*a-1)*dx*yetComputed[idxSrc3] + FReal((a-1)*(a-1))*yetComputed[idxSrc4])/dist2;
                        if(a+b < 2*P){
                            //Computation of derivatives Psi_{a,b,1}
                            // |x-y|^2 * Psi_{a,b,1} + 2*b*dy*Psi_{a,b-1,1} + b*(b-1)*Psi_{a,b-2,1} + 2*a*dx*Psi_{a-1,b,1} + a*(a-1)*Psi_{a-2,b,1} + dz*Psi_{a,b,0}= 0
                            idxTarget = powerToIdx(a,b,1);
                            idxSrc1 = powerToIdx(a,b-1,1);
                            idxSrc2 = powerToIdx(a,b-2,1);
                            idxSrc3 = powerToIdx(a-1,b,1);
                            idxSrc4 = powerToIdx(a-2,b,1);
                            idxSrc5 = powerToIdx(a,b,0);
                            yetComputed[idxTarget] = -(FReal(2*b)*dy*yetComputed[idxSrc1] + FReal(b*(b-1))*yetComputed[idxSrc2]
                                                       + FReal(2*a)*dx*yetComputed[idxSrc3] + FReal(a*(a-1))*yetComputed[idxSrc4]
                                                       + dz*yetComputed[idxSrc5])/dist2;
                        }
                        for(c=2 ; c <= 2*P-b-a ; ++c){
                            //Computation of derivatives Psi_{a,b,c} with a >= 2
                            // |x-y|^2*Psi_{a,b,c} + (2*a-1)*dx*Psi_{a-1,b,c} + a*(a-2)*Psi_{a-2,b,c} + 2*b*dy*Psi_{a,b-1,c} + b*(b-1)*Psi_{a,b-2,c}
                            // + 2*c*dz*Psi_{a,b,c-1}} = 0
                            idxTarget = powerToIdx(a,b,c);
                            idxSrc1 = powerToIdx(a-1,b,c);
                            idxSrc2 = powerToIdx(a,b-1,c);
                            idxSrc3 = powerToIdx(a,b,c-1);
                            idxSrc4 = powerToIdx(a-2,b,c);
                            idxSrc5 = powerToIdx(a,b-2,c);
                            idxSrc6 = powerToIdx(a,b,c-2);
                            yetComputed[idxTarget] = -(FReal(2*a-1)*dx*yetComputed[idxSrc1] + FReal((a-1)*(a-1))*yetComputed[idxSrc4]
                                                       + FReal(2*b)*dy*yetComputed[idxSrc2] + FReal(b*(b-1))*yetComputed[idxSrc5]
                                                       + FReal(2*c)*dz*yetComputed[idxSrc3] + FReal(c*(c-1))*yetComputed[idxSrc6])/dist2;
                        }
                    }
                }
            }
        }
    }


    /////////////////////////////////
    ///////// Public Methods ////////
    /////////////////////////////////

public:

    /*Constructor, need system information*/
    FTaylorKernel(const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter) :
        boxWidth(inBoxWidth),
        treeHeight(inTreeHeight),
        widthAtLeafLevel(inBoxWidth/FReal(1 << (inTreeHeight-1))),
        widthAtLeafLevelDiv2(widthAtLeafLevel/2),
        boxCorner(inBoxCenter.getX()-(inBoxWidth/2),inBoxCenter.getY()-(inBoxWidth/2),inBoxCenter.getZ()-(inBoxWidth/2))
    {
        FReal facto;
        this->precomputeFactorials() ;
        for(int i=0, a = 0, b = 0, c = 0; i<SizeVector ; ++i)
        {
            facto = static_cast<FReal>(fact3int(a,b,c));
            _coeffPoly[i] = FReal(1.0)/(facto);
#ifdef OC
            _coeffPoly[i] = static_cast<FReal>(factorials[a+b+c])/(facto*facto);
#endif
            this->incPowers(&a,&b,&c);       //inc powers of expansion
        }
    }

    /* Default destructor
   */
    virtual ~FTaylorKernel(){

    }

    /**P2M
   * @brief Fill the Multipole with the field created by the cell
   * particles.
   *
   * Formula :
   * \f[
   *   M_{k} = \frac{|k|!}{k! k!} \sum_{j=0}^{N}{ q_j   (x_c-x_j)^{k}}
   * \f]
   * where \f$x_c\f$ is the centre of the cell and \f$x_j\f$ the \f$j^{th}\f$ particles and \f$q_j\f$ its charge and  \f$N\f$ the particle number.
   */
    void P2M(CellClass* const pole,
             const ContainerClass* const particles) override
    {
        //Copying cell center position once and for all
        const FPoint<FReal>& cellCenter = getLeafCenter(pole->getCoordinate());
        FReal * FRestrict multipole = pole->getMultipole();
        FMemUtils::memset(multipole,0,SizeVector*sizeof(FReal(0.0)));
        FReal multipole2[SizeVector] ;

        // Iterator over Particles
        FSize nbPart = particles->getNbParticles(), i;
        const FReal* const * positions = particles->getPositions();
        const FReal* posX = positions[0];
        const FReal* posY = positions[1];
        const FReal* posZ = positions[2];

        const FReal* phyValue = particles->getPhysicalValues();

        // Iterating over Particles
        FReal xc = cellCenter.getX(), yc = cellCenter.getY(), zc = cellCenter.getZ() ;
        FReal dx[3] ;
        for(FSize idPart=0 ; idPart<nbPart ; ++idPart){
            dx[0]         = xc - posX[idPart] ;
            dx[1]         = yc - posY[idPart] ;
            dx[2]         = zc - posZ[idPart] ;
            multipole2[0] = phyValue[idPart]  ;

            int leading[3] = {0,0,0 } ;
            for (int k=1, t=1, tail=1; k <= P; ++k, tail=t)
            {
                for (i = 0; i < 3; ++i)
                {
                    int head = leading[i];
                    leading[i] = t;
                    for ( int j = head; j < tail; ++j, ++t)
                    {
                        multipole2[t] = multipole2[j] *dx[i] ;
                    }
                } // for i
            }// for k

            //is that a possible saxpy ?
            for( i=0 ; i < SizeVector ; ++i)
            {
                multipole[i] +=  multipole2[i] ;
                multipole2[i] = 0.0;
            }
        }  // end loop on particles

        // Multiply by the coefficient
        for( i=0 ; i < SizeVector ; ++i)
        {
            multipole[i] *=_coeffPoly[i] ;
        }
    }


    /**
   * @brief Fill the parent multipole with the 8 values of child multipoles
   *
   *
   */
    void M2M(CellClass* const FRestrict pole,
             const CellClass*const FRestrict *const FRestrict child,
             const int inLevel) override
    {
        int a=0,b=0,c=0;

        //Indexes of powers
        int idx_a,idx_b,idx_c;

        //Distance from current child to parent
        FReal dx = 0.0;
        FReal dy = 0.0;
        FReal dz = 0.0;
        //Center point of parent cell
        const FPoint<FReal>& cellCenter = getCellCenter(pole->getCoordinate(),inLevel);
        FReal * FRestrict mult = pole->getMultipole();

        //Iteration over the eight children
        int idxChild;
        FReal coeff;
        for(idxChild=0 ; idxChild<8 ; ++idxChild)
        {
            if(child[idxChild]){ //Test if child exists
                const FPoint<FReal>& childCenter = getCellCenter(child[idxChild]->getCoordinate(),inLevel+1);
                const FReal * FRestrict multChild = child[idxChild]->getMultipole();

                //Set the distance between centers of cells
                dx = cellCenter.getX() - childCenter.getX();
                dy = cellCenter.getY() - childCenter.getY();
                dz = cellCenter.getZ() - childCenter.getZ();

                // Precompute the  arrays of dx^i
                arrayDX[0] = 1.0 ;
                arrayDY[0] = 1.0 ;
                arrayDZ[0] = 1.0 ;
                for (int i = 1 ; i <= P ; ++i) 	{
                    arrayDX[i] = dx * arrayDX[i-1] ;
                    arrayDY[i] = dy * arrayDY[i-1] ;
                    arrayDZ[i] = dz * arrayDZ[i-1] ;
                }

                a=0;
                b=0;
                c=0;
                FReal value;

                //Iteration over parent multipole array
                for(int idxMult = 0 ; idxMult<SizeVector ; idxMult++)
                {
                    value = 0.0;
                    int idMultiChild;

                    //Iteration over the powers to find the cell multipole
                    //involved in the computation of the parent multipole
                    for(idx_a=0 ; idx_a <= a ; ++idx_a){
                        for(idx_b=0 ; idx_b <= b ; ++idx_b){
                            for(idx_c=0 ; idx_c <= c ; ++idx_c){

                                //Computation
                                //Child multipole involved
                                idMultiChild = powerToIdx(a-idx_a,b-idx_b,c-idx_c);
                                coeff = FReal(1.0)/fact3int(idx_a,idx_b,idx_c);
                                value += multChild[idMultiChild]*coeff*arrayDX[idx_a]*arrayDY[idx_b]*arrayDZ[idx_c];
                            }
                        }
                    }
                    mult[idxMult] += value;
                    incPowers(&a,&b,&c);
                }
            }
        }
    }

    /**
   *@brief Convert the multipole expansion into local expansion The
   * operator do not use symmetries.
   *
   * Formula : \f[ L_{\mathbf{n}}^{c} = \frac{|n|!}{n! n!}
   * \sum_{\mathbf{k}=0}^{p} \left [ M_\mathbf{k}^c \,
   * \Psi_{\mathbf{,n+k}}( \mathbf{x}_c^{target})\right ] \f]
   * and \f[ \Psi_{\mathbf{,i}}^{c}(\mathbf{x}) =
   * \frac{\partial^i}{\partial x^i} \frac{1}{|x-x_c^{src}|} =  \frac{\partial^{i_1}}{\partial x_1^{i_1}} \frac{\partial^{i_2}}{\partial x_2^{i_2}} \frac{\partial^{i_3}}{\partial x_3^{i_3}} \frac{1}{|x-x_c^{src}|}\f]
   *
   * Where \f$x_c^{src}\f$ is the centre of the cell where the
   * multiplole are considered,\f$x_c^{target}\f$ is the centre of the
   * current cell. The cell where we compute the local expansion.
   *
   */
    void M2L(CellClass* const FRestrict inLocal, const CellClass* distantNeighbors[],
             const int /*neighborPositions*/[], const int inSize, const int inLevel)  override {
        //Iteration over distantNeighbors
        const FPoint<FReal> & locCenter = getCellCenter(inLocal->getCoordinate(),inLevel);
        FReal * FRestrict iterLocal = inLocal->getLocal();

        for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
            const FPoint<FReal> curDistCenter = getCellCenter(distantNeighbors[idxExistingNeigh]->getCoordinate(),inLevel);
            FMemUtils::memset(this->_PsiVector,0,sizeDerivative*sizeof(FReal(0.0)));

            // Compute derivatives on  locCenter - curDistCenter
            //                           target       source
            FReal dx = locCenter.getX() - curDistCenter.getX();
            FReal dy = locCenter.getY() - curDistCenter.getY();
            FReal dz = locCenter.getZ() - curDistCenter.getZ();


            //Iterative Computation of all the derivatives needed
            this->computeFullDerivative(dx,dy,dz,this->_PsiVector);
            const FReal * multipole = distantNeighbors[idxExistingNeigh]->getMultipole();

            int al=0,bl=0 /*,cl=0*/ ;   // For local array
            int am,bm,cm;               // For distant array
            int id=0;                   // Index of iterLocal
            int i;

            //Case (al,bl,cl) == (0,0,0)
            for(int j = 0 ; j < SizeVector ; ++j){ //corresponding powers am,bm,cm
                iterLocal[0] += this->_PsiVector[j]*multipole[j];
            }


            for(i=1, id=1 ; i<=P ; ++i){
                //Case (i,0,0) :
                FReal fctl   = factorials[i];
                FReal coeffL = FReal(1.0)/(fctl);
                //Iterator over multipole array
                FReal tmp = 0.0;
                am=0;	  bm=0;  cm=0;
                for(int j = 0 ; j < SizeVector ; ++j){ //corresponding powers am,bm,cm
                    int idxPsi = powerToIdx(i+am,bm,cm);
                    tmp += this->_PsiVector[idxPsi]*multipole[j];
                    incPowers(&am,&bm,&cm);
                }
                iterLocal[id] += tmp*coeffL ; //access to iterLocal[ (i,0,0) ] is inlined.

                //Case (i-1,1,0) :
                fctl = factorials[i-1];
                coeffL = FReal(1.0)/(fctl);
                //Iterator over multipole array
                tmp = 0.0;
                am=0;	  bm=0;  cm=0;
                for(int j = 0 ; j < SizeVector ; ++j){ //corresponding powers am,bm,cm
                    int idxPsi = powerToIdx((i-1)+am,1+bm,cm);
                    tmp += this->_PsiVector[idxPsi]*multipole[j];
                    //updating a,b,c
                    incPowers(&am,&bm,&cm);
                }
                iterLocal[++id] += tmp*coeffL; //access to iterLocal[ (i-1,1,0) ] is inlined


                //Case (i-1,0,1) :
                //Values of fctl and coeffL are the same for (i-1,1,0) and (i-1,0,1)
                //Iterator over multipole array
                tmp = 0.0;
                am=0;	  bm=0;  cm=0;
                for(int j = 0 ; j < SizeVector ; ++j){ //corresponding powers am,bm,cm
                    int idxPsi = powerToIdx((i-1)+am,bm,1+cm);
                    tmp += this->_PsiVector[idxPsi]*multipole[j];
                    //updating a,b,c
                    incPowers(&am,&bm,&cm);
                }
                iterLocal[++id] += tmp*coeffL; //access to iterLocal[ (i-1,0,1) ] is inlined

                //End of specific case
                //Start of general case :
                ++id;
                for(al=i-2 ; al>= 0 ; --al)
                {
                    for(bl=i-al ; bl>=0 ; --bl, ++id)
                    {
                        am = 0 ;
                        bm = 0 ;
                        cm = 0 ;
                        tmp = FReal(0);
                        fctl = fact3int(al,bl,i-(al+bl));
                        coeffL = FReal(1.0)/(fctl);
                        for(int j = 0 ; j < SizeVector ; ++j){ //corresponding powers am,bm,cm
                            int idxPsi = powerToIdx(al+am,bl+bm,(i-al-bl)+cm);
                            tmp += this->_PsiVector[idxPsi]*multipole[j];
                            //updating a,b,c
                            incPowers(&am,&bm,&cm);
                        }
                        iterLocal[id] += tmp*coeffL;
                    }
                }
            }
        }
    }


    /**
   *@brief Translate the local expansion of parent cell to child cell
   *
   * One need to translate the local expansion on a father cell
   * centered in \f$\mathbf{x}_p\f$ to its eight daughters centered in
   * \f$\mathbf{x}_p\f$ .
   *
   * Local expansion for the daughters will be :
   * \f[ \sum_{\mathbf{k}=0}^{|k|<P} L_k * (\mathbf{x}-\mathbf{x}_f)^{\mathbf{k}} \f]
   * with :
   *
   *\f[ L_{n} = \sum_{\mathbf{k}=\mathbf{n}}^{|k|<P} C_{\mathbf{k}}^{\mathbf{n}} * (\mathbf{x}_f-\mathbf{x}_p)^{\mathbf{k}-\mathbf{n}} \f]
   */

    void L2L(const CellClass* const FRestrict fatherCell,
             CellClass* FRestrict * const FRestrict childCell,
             const int inLevel) override
    {
        FPoint<FReal> locCenter = getCellCenter(fatherCell->getCoordinate(),inLevel);

        // Get father local expansion
        const FReal* FRestrict fatherExpansion = fatherCell->getLocal()  ;
        int idxFatherLoc;               //index of Father local expansion to be read.
        FReal dx,  dy,  dz, coeff;
        int ap, bp, cp, af, bf, cf;     //Indexes of expansion for father and child.

        // For all children
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){

            if(childCell[idxChild]){ //test if child exists

                FReal* FRestrict childExpansion = childCell[idxChild]->getLocal() ;
                const FPoint<FReal>& childCenter = getCellCenter(childCell[idxChild]->getCoordinate(),inLevel+1);

                //Set the distance between centers of cells
                // Child - father
                dx = childCenter.getX()-locCenter.getX();
                dy = childCenter.getY()-locCenter.getY();
                dz = childCenter.getZ()-locCenter.getZ();
                // Precompute the  arrays of dx^i
                arrayDX[0] = 1.0 ;
                arrayDY[0] = 1.0 ;
                arrayDZ[0] = 1.0 ;
                for (int i = 1 ; i <= P ; ++i) 	{
                    arrayDX[i] = dx * arrayDX[i-1] ;
                    arrayDY[i] = dy * arrayDY[i-1] ;
                    arrayDZ[i] = dz * arrayDZ[i-1] ;
                }

                //iterator over child's local expansion (to be filled)
                af=0;	bf=0;	cf=0;
                for(int k=0 ; k<SizeVector ; ++k){

                    //Iterator over parent's local array
                    for(ap=af ; ap<=P ; ++ap)
                    {
                        for(bp=bf ; bp<=P ; ++bp)
                        {
                            for(cp=cf ; ((cp<=P) && (ap+bp+cp) <= P) ; ++cp)
                            {
                                idxFatherLoc = powerToIdx(ap,bp,cp);
                                coeff = combin(ap,af) * combin(bp,bf) * combin(cp,cf);
                                childExpansion[k] += coeff*fatherExpansion[idxFatherLoc]*arrayDX[ap-af]*arrayDY[bp-bf]*arrayDZ[cp-cf] ;
                            }
                        }
                    }
                    incPowers(&af,&bf,&cf);
                }
            }
        }
    }




    /**L2P
   *@brief Apply on the particles the force computed from the local expansion to all particles in the cell
   *
   *
   * Formula :
   * \f[
   *   Potential = \sum_{j=0}^{nb_{particles}}{q_j \sum_{k=0}^{P}{ L_k * (x_j-x_c)^{k}}}
   * \f]
   *
   * where \f$x_c\f$ is the centre of the local cell and \f$x_j\f$ the
   * \f$j^{th}\f$ particles and \f$q_j\f$ its charge.
   */
    void L2P(const CellClass* const local,
             ContainerClass* const particles) override
    {
        FPoint<FReal> locCenter = getLeafCenter(local->getCoordinate());
        //Iterator over particles
        FSize nbPart = particles->getNbParticles();

        //Iteration over Local array
        //
        const FReal * iterLocal = local->getLocal();
        const FReal * const * positions = particles->getPositions();
        const FReal * posX = positions[0];
        const FReal * posY = positions[1];
        const FReal * posZ = positions[2];

        FReal * forceX = particles->getForcesX();
        FReal * forceY = particles->getForcesY();
        FReal * forceZ = particles->getForcesZ();
        FReal * targetsPotentials = particles->getPotentials();
        FReal * phyValues = particles->getPhysicalValues();

        //Iteration over particles
        for(FSize i=0 ; i<nbPart ; ++i){
            //
            FReal dx =  posX[i] - locCenter.getX();
            FReal dy =  posY[i] - locCenter.getY();
            FReal dz =  posZ[i] - locCenter.getZ();
            // Precompute an arrays of Array[i] = dx^(i-1)
            arrayDX[0] = 0.0 ;
            arrayDY[0] = 0.0 ;
            arrayDZ[0] = 0.0 ;

            arrayDX[1] = 1.0 ;
            arrayDY[1] = 1.0 ;
            arrayDZ[1] = 1.0 ;

            for (int d = 2 ; d <= P+1 ; ++d){ //Array is staggered : Array[i] = dx^(i-1)
                arrayDX[d] = dx * arrayDX[d-1] ;
                arrayDY[d] = dy * arrayDY[d-1] ;
                arrayDZ[d] = dz * arrayDZ[d-1] ;
            }
            FReal partPhyValue = phyValues[i];
            //
            FReal  locPot = 0.0, locForceX = 0.0, locForceY = 0.0, locForceZ = 0.0 ;
            int a=0,b=0,c=0;
            for(int j=0 ; j<SizeVector ; ++j){
                FReal locForce     = iterLocal[j];
                // compute the potential
                locPot += iterLocal[j]*arrayDX[a+1]*arrayDY[b+1]*arrayDZ[c+1];
                //Application of forces
                locForceX += FReal(a)*locForce*arrayDX[a]*arrayDY[b+1]*arrayDZ[c+1];
                locForceY += FReal(b)*locForce*arrayDX[a+1]*arrayDY[b]*arrayDZ[c+1];
                locForceZ += FReal(c)*locForce*arrayDX[a+1]*arrayDY[b+1]*arrayDZ[c];
                incPowers(&a,&b,&c);
            }
            targetsPotentials[i]  +=/* partPhyValue*/locPot ;
            forceX[i]             += partPhyValue*locForceX ;
            forceY[i]             += partPhyValue*locForceY ;
            forceZ[i]             += partPhyValue*locForceZ ;
        }
    }

    /** P2P
      * This function proceed the P2P using particlesMutualInteraction
      * The computation is done for interactions with an index <= 13.
      * (13 means current leaf (x;y;z) = (0;0;0)).
      * Calling this method in multi thread should be done carrefully.
      */
    void P2P(const FTreeCoordinate& inPosition,
             ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict inSources,
             ContainerClass* const inNeighbors[], const int neighborPositions[],
             const int inSize) override {
        if(inTargets == inSources){
            FP2PRT<FReal>::template Inner<ContainerClass>(inTargets);
            P2POuter(inPosition, inTargets, inNeighbors, neighborPositions, inSize);
        }
        else{
            const ContainerClass* const srcPtr[1] = {inSources};
            FP2PRT<FReal>::template FullRemote<ContainerClass>(inTargets,srcPtr,1);
            FP2PRT<FReal>::template FullRemote<ContainerClass>(inTargets,inNeighbors,inSize);
        }
    }

    void P2POuter(const FTreeCoordinate& /*inLeafPosition*/,
             ContainerClass* const FRestrict inTargets,
             ContainerClass* const inNeighbors[], const int neighborPositions[],
             const int inSize) override {
        int nbNeighborsToCompute = 0;
        while(nbNeighborsToCompute < inSize
              && neighborPositions[nbNeighborsToCompute] < 14){
            nbNeighborsToCompute += 1;
        }
        FP2PRT<FReal>::template FullMutual<ContainerClass>(inTargets,inNeighbors,nbNeighborsToCompute);
    }


    /** Use mutual even if it not useful and call particlesMutualInteraction */
    void P2PRemote(const FTreeCoordinate& /*inPosition*/,
                   ContainerClass* const FRestrict inTargets, const ContainerClass* const FRestrict /*inSources*/,
                   const ContainerClass* const inNeighbors[], const int neighborPositions[],
                   const int inSize) override {
        FP2PRT<FReal>::template FullRemote<ContainerClass>(inTargets,inNeighbors,inSize);
    }

};

#endif
