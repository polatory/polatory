#ifndef FTESTOPENCLCODE_HPP
#define FTESTOPENCLCODE_HPP

#include "../../Utils/FGlobal.hpp"
#include "../StarPUUtils/FStarPUDefaultAlign.hpp"
#include "../OpenCl/FTextReplacer.hpp"

// Initialize the types
template <class FReal>
class FTestOpenCLCode{
    FTextReplacer kernelfile;
    size_t dim;

public:
    //FTestOpenCLCode() : kernelfile("/home/berenger/Projets/ScalfmmGit/scalfmm/Src/GroupTree/OpenCl/FEmptyKernel.cl"){
    FTestOpenCLCode() : kernelfile("../Src/GroupTree/TestKernel/FTestKernel.cl"){
        if(sizeof(FReal) == sizeof(double)){
            kernelfile.replaceAll("___FReal___", "double");
        }
        else{
            kernelfile.replaceAll("___FReal___", "float");
        }
        FAssertLF((typeid(FSize) == typeid(long long int)));
        kernelfile.replaceAll("___FSize___", "long long int");
        kernelfile.replaceAll("___FParticleValueClass___", "long long");
        kernelfile.replaceAll("___NbSymbAttributes___", 0);
        kernelfile.replaceAll("___NbAttributesPerParticle___", 1);
        const size_t structAlign = FStarPUDefaultAlign::StructAlign;
        kernelfile.replaceAll("___DefaultStructAlign___", structAlign);
        kernelfile.replaceAll("___FP2PDefaultAlignement___", FP2PDefaultAlignement);

        dim = 1;
    }

    const char* getKernelCode(const int /*inDevId*/){
        return kernelfile.getContent();
    }

    void releaseKernelCode(){
        kernelfile.clear();
    }

    unsigned int getNbDims() const {
        return 1;
    }

    const size_t* getNbGroups(const int /*inSizeInterval*/) const {
        // We return 1
        return &dim;
    }

    const size_t* getGroupSize() const {
        // We return 1
        return &dim;
    }
};


#endif // FTESTOPENCLCODE_HPP

