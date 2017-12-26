// See LICENCE file at project root

#ifndef FTAYLORFLOPSKERNEL_HPP
#define FTAYLORFLOPSKERNEL_HPP



/**
 * @author Cyrille Piacibello
 * @class FTaylorFlopsKernel
 * @brief
 * Please read the license
 *
 *
 * @brief This kernel provide the flops needed for each of the
 * operators used in the Taylor Kernel.
 */
template<  class FReal, class CellClass, class ContainerClass, int P, int order>
class FTaylorFlopsKernel : public FAbstractKernels<CellClass,ContainerClass> {
    
    static const int SizeVector = ((P+1)*(P+2)*(P+3))*order/6;
    static const int sizeDerivative = ((2*P+1)*(P+1)*(P+3))*order/3;
    unsigned int incPowersFlop;
    const unsigned treeHeight;
    unsigned int pTIdx;
    //Flops for each functions:
    unsigned long long flopsP2M, flopsM2M, flopsM2L, flopsL2L, flopsL2P, flopsP2P;

    unsigned long long *flopsPerLevelM2M, *flopsPerLevelM2L, *flopsPerLevelL2L;


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



public:
    FTaylorFlopsKernel(const int inTreeHeight, const FReal inBoxWidth, const FPoint<FReal>& inBoxCenter) :
        treeHeight(inTreeHeight),
        flopsP2M(0),
        flopsM2M(0),
        flopsM2L(0),
        flopsL2L(0),
        flopsL2P(0),
        flopsP2P(0),
        pTIdx(10),
        incPowersFlop(10),
        flopsPerLevelM2M(NULL),
        flopsPerLevelM2L(NULL),
        flopsPerLevelL2L(NULL)
    {
        pTIdx = 10;
        flopsPerLevelM2M = new unsigned long long [inTreeHeight];
        flopsPerLevelM2L = new unsigned long long [inTreeHeight];
        flopsPerLevelL2L = new unsigned long long [inTreeHeight];
        for (unsigned int level = 0; level<inTreeHeight; ++level){
            flopsPerLevelM2M[level] = flopsPerLevelM2L[level] = flopsPerLevelL2L[level] = 0;
        }
    }

    virtual ~FTaylorFlopsKernel(){
        std::cout << "\n=================================================="
                  << "\n- Flops for P2M = " << flopsP2M
                  << "\n- Flops for M2M = " << flopsM2M
                  << "\n- Flops for M2L = " << flopsM2L
                  << "\n- Flops for L2L = " << flopsL2L
                  << "\n- Flops for L2P = " << flopsL2P
                  << "\n- Flops for P2P = " << flopsP2P
                  << "\n- Overall Flops = " << flopsP2M + flopsM2M + flopsM2L + flopsL2L + flopsL2P + flopsP2P
                  << "\n==================================================\n"
                  << std::endl;

        std::cout << "\n=================================================="
                  << "\n- Flops for P2M/M2M" << std::endl;
        for (unsigned int level=0; level<treeHeight; ++level)
            if (level < treeHeight-1)
                std::cout << "  |- at level " << level << " flops = " << flopsPerLevelM2M[level] << std::endl;
            else
                std::cout << "  |- at level " << level << " flops = " << flopsP2M << std::endl;
        std::cout << "=================================================="
                  << "\n- Flops for M2L" << std::endl;
        for (unsigned int level=0; level<treeHeight; ++level)
            std::cout << "  |- at level " << level << " flops = " << flopsPerLevelM2L[level] << std::endl;
        std::cout << "=================================================="
                  << "\n- Flops for L2L/L2P" << std::endl;
        for (unsigned int level=0; level<treeHeight; ++level)
            if (level < treeHeight-1)
                std::cout << "  |- at level " << level << " flops = " << flopsPerLevelL2L[level] << std::endl;
            else
                std::cout << "  |- at level " << level << " flops = " << flopsL2P << std::endl;
        std::cout << "==================================================" << std::endl;
        if (flopsPerLevelM2M) delete [] flopsPerLevelM2M;
        if (flopsPerLevelM2L) delete [] flopsPerLevelM2L;
        if (flopsPerLevelL2L) delete [] flopsPerLevelL2L;

    }

    void P2M(CellClass* const pole,
             const ContainerClass* const particles)override
    {
        //Nb_Particule * [3 (: dx,dy,dz) + SizeVector*4 (: Classic operations) ] + SizeVector (: multiply multipole by coefficient)
        this->flopsP2M += (particles->getNbParticles())*(3+SizeVector*4)+SizeVector;
    }

    void M2M(CellClass* const FRestrict pole,
             const CellClass*const FRestrict *const FRestrict child,
             const int inLevel)override
    {
        //Powers of expansions
        int a=0,b=0,c=0;

        //Indexes of powers
        int idx_a,idx_b,idx_c;

        unsigned int flops = 8;
        for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex)
        {
            if (child[ChildIndex])
            {
                a=0; b=0; c=0;
                flops += 6 + 6*P;
                for(int k=0 ; k<SizeVector ; ++k)
                {
                    for(idx_a=0 ; idx_a <= a ; ++idx_a){
                        for(idx_b=0 ; idx_b <= b ; ++idx_b){
                            for(idx_c=0 ; idx_c <= c ; ++idx_c){
                                flops += pTIdx;
                                flops += 30;
                            }
                        }
                    }
                    incPowers(&a,&b,&c);
                    flops+=incPowersFlop;
                }
            }
        }
        flopsM2M += flops;
        flopsPerLevelM2M[inLevel] += flops;
    }

    void M2L(CellClass* const FRestrict local,             // Target cell
             const CellClass* distantNeighbors[],       // Sources to be read
    const int /*positions*/[],
    const int size, const int inLevel)override
    {
        unsigned int flops = 343;
        for (unsigned int idx=0; idx<size; ++idx)
        {
                flops += 2+SizeVector*(2+SizeVector+2);
        }
        flopsM2L += flops;
        flopsPerLevelM2L[inLevel] += flops;
    }

    void L2L(const CellClass* const FRestrict fatherCell,
             CellClass* FRestrict * const FRestrict childCell,
             const int inLevel)override
    {
        int ap, bp, cp, af, bf, cf;     //Indexes of expansion for father and child.
        unsigned int flops = 2;
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            // if child exists
            if(childCell[idxChild]){
                af=0;	bf=0;	cf=0;
                flops += 6 + 6*P;
                for(int k=0 ; k<SizeVector ; ++k)
                {
                    //Iterator over parent's local array
                    for(ap=af ; ap<=P ; ++ap)
                    {
                        for(bp=bf ; bp<=P ; ++bp)
                        {
                            for(cp=cf ; ((cp<=P) && (ap+bp+cp) <= P) ; ++cp)
                            {
                                flops += pTIdx;
                                flops += 2+3*2;/*3 calls to combin + multiplication of the 3 results*/
                                flops += 5;
                            }
                        }
                    }
                    incPowers(&af,&bf,&cf);
                    flops += incPowersFlop;
                }
            }
        }
        flopsL2L += flops;
        flopsPerLevelL2L[inLevel] += flops;
    }

    void L2P(const CellClass* const local,
             ContainerClass* const particles)override
    {
        unsigned int flops = 12;
        FSize nbPart = particles->getNbParticles();
        for(int i=0 ; i<nbPart ; ++i){
            flops += 16 + 3*(P-1);/*Distances computation*/
            for(int j=0 ; j<SizeVector ; ++j){
                flops += 4+3*4;
                flops += incPowersFlop;
            }
            flops += 1 + 3*2; /*store of results*/
        }
        flopsL2P += flops;
    }

    void P2P(const FTreeCoordinate& /*inLeafPosition*/,
             ContainerClass* const FRestrict targets, const ContainerClass* const FRestrict /*sources*/,
             ContainerClass* const directNeighborsParticles[], const int /*positions*/, const int /*size*/)override
    {}
    
};

#endif
