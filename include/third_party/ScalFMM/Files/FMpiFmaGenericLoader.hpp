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
  FMpiFmaGenericLoader(const std::string inFilename,const FMpi::FComm& comm, const bool useMpiIO = false)
    : FFmaGenericLoader<FReal>(inFilename,true),myNbOfParticles(0),idxParticles(0),headerSize(0)
  {
    FSize startPart = comm.getLeft(nbParticles);
    FSize endPart   = comm.getRight(nbParticles);
    this->start = startPart;
    this->myNbOfParticles = endPart-startPart;
    std::cout << "Proc " << comm.processId() << " will hold " << myNbOfParticles << std::endl;
    
    //This is header size in bytes
    //   MEANING :      sizeof(FReal)+nbAttr, nb of parts, boxWidth+boxCenter
    headerSize = sizeof(int)*2 + sizeof(FSize) + sizeof(FReal)*4;
    //To this header size, we had the parts that belongs to proc on my left
    file->seekg(headerSize + startPart*typeData[1]*sizeof(FReal));
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
