#ifndef _FMPISPLITFMALOADER_HPP_
#define _FMPISPLITFMALOADER_HPP_

#include "Files/FFmaGenericLoader.hpp"

template< typename FReal>
class FMpiSplitFmaLoader : public FFmaGenericLoader<FReal> {

    FSize _particleCount = 0;


public: 
    FMpiSplitFmaLoader(const std::string& filename, int procRank) : 
        FFmaGenericLoader<FReal>(
            std::string(filename).replace(
                filename.rfind(".main."),
                6,
                "."+std::to_string(procRank)+".") ) {

        FFmaGenericLoader<FReal> mainLoader(filename);
        _particleCount = mainLoader.getNumberOfParticles();
    } 

    FSize getNumberOfParticles() const {
        return _particleCount;
    }

    FSize getMyNumberOfParticles() const {
        return FFmaGenericLoader<FReal>::getNumberOfParticles();
    }

    FSize getStart() const {
        return 0;
    }

};


#endif
