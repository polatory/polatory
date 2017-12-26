// See LICENCE file at project root

// ==== CMAKE =====
// @FUSE_MPI
// ================


#ifndef FMPIFMAGENERICLOADER_HPP
#define FMPIFMAGENERICLOADER_HPP


#include "Utils/FMpi.hpp"
#include "Files/FFmaGenericLoader.hpp"

template <class FReal>
class FMpiFmaGenericLoader : public FFmaGenericLoader<FReal> {
protected:
    using FFmaGenericLoader<FReal>::nbParticles;
    using FFmaGenericLoader<FReal>::file;
    using FFmaGenericLoader<FReal>::typeData;

  FSize myNbOfParticles;     //Number of particles that the calling process will manage
  MPI_Offset idxParticles;   //
  FSize start;               // number of my first parts in file
  size_t headerSize;
public:
    FMpiFmaGenericLoader(const std::string inFilename,const FMpi::FComm& comm, const bool /*useMpiIO*/ = false)
        : FFmaGenericLoader<FReal>(inFilename),myNbOfParticles(0),idxParticles(0),headerSize(0)
  {
    FSize startPart = comm.getLeft(nbParticles);
    FSize endPart   = comm.getRight(nbParticles);
    this->start = startPart;
    this->myNbOfParticles = endPart-startPart;
    std::cout << " startPart "<< startPart << " endPart " << endPart<<std::endl;
    std::cout << "Proc " << comm.processId() << " will hold " << myNbOfParticles << std::endl;

    if(this->binaryFile) {
        //This is header size in bytes
        //   MEANING :      sizeof(FReal)+nbAttr, nb of parts, boxWidth+boxCenter
        headerSize = sizeof(int)*2 + sizeof(FSize) + sizeof(FReal)*4;
        //To this header size, we had the parts that belongs to proc on my left
        file->seekg(headerSize + startPart*typeData[1]*sizeof(FReal));
    } else {
        // First finish to read the current line
        file->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        for(int i = 0; i < startPart; ++i) {
            file->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
  }

  ~FMpiFmaGenericLoader(){
  }

  FSize getMyNumberOfParticles() const{
    return myNbOfParticles;
  }

  FSize getStart() const{
    return start;
  }

  /**
   * Given an index, get the one particle from this index
   */
  void fill1Particle(FReal*datas,FSize indexInFile){
    file->seekg(headerSize+(int(indexInFile)*FFmaGenericLoader<FReal>::getNbRecordPerline()*sizeof(FReal)));
    file->read((char*) datas,FFmaGenericLoader<FReal>::getNbRecordPerline()*sizeof(FReal));
  }

};

#endif //FMPIFMAGENERICLOADER_HPP
