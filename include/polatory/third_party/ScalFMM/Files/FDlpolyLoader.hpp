// See LICENCE file at project root
#ifndef FDlpolyLOADER_HPP
#define FDlpolyLOADER_HPP


#include <iostream>
#include <fstream>
#include <limits>

#include "../Utils/FGlobal.hpp"
#include "FAbstractLoader.hpp"
#include "../Utils/FPoint.hpp"

template <class FReal>
class FDlpolyLoader : public FAbstractLoader<FReal> {
protected:
//    std::ifstream file;         //< The file to read
    FPoint<FReal>        centerOfBox;          //< The center of box read from file
    FReal         boxWidth;             //< the box width read from file
    int           nbParticles;            //< the number of particles read from file
    int           levcfg  ;               //< DL_POLY CONFIG file key. 0,1 or 2
    FReal         energy  ;
protected:
    enum Type{
        OW,
        HW,
        Na,
        Cl,
        Undefined,
    };
public:
    virtual ~FDlpolyLoader()
    {}
    virtual bool isOpen() const =0;
    virtual void fillParticle(FPoint<FReal>* inPosition, FReal inForces[3], FReal* inPhysicalValue, int* inIndex)=0;

    /**
       * To get the number of particles from this loader
       * @param the number of particles the loader can fill
       */
     FSize getNumberOfParticles() const{
         return FSize(this->nbParticles);
     }

     /**
       * The center of the box from the simulation file opened by the loader
       * @return box center
       */
     FPoint<FReal> getCenterOfBox() const{
         return this->centerOfBox;
     }

     /**
       * The box width from the simulation file opened by the loader
       * @return box width
       */
     FReal getBoxWidth() const{
         return this->boxWidth;
     }

     FReal getEnergy() const{
       return this->energy;
     }
     void getPhysicalValue(std::string &type,FReal & outPhysicalValue, int & outIndex) const{
      	if( strncmp(type.c_str(), "OW", 2) == 0){
      		outPhysicalValue = FReal(-0.82);
      		outIndex = OW;
      	}
      	else if( strncmp(type.c_str(), "HW", 2) == 0){
      		outPhysicalValue = FReal(-0.41);
      		outIndex = HW;
      	}
      	else if( strncmp(type.c_str(), "Na", 2) == 0){
      		outPhysicalValue = FReal(1.0);
      		outIndex = Na;
      	}
      	else if( strncmp(type.c_str(), "Cl", 2) == 0){
      		outPhysicalValue = FReal(-1.0);
      		outIndex = Cl;
      	}
      	else{

      		std::cerr << "Atom type not defined "<< type << std::endl;
      		exit(-1);
      	}
    //  	std::cout << "Atom type " << type << "  "<< outPhysicalValue << " " << outIndex <<std::endl;
      }

     void getPhysicalValue(char type[2],FReal & outPhysicalValue, int & outIndex) const{
     	if( strncmp(type, "OW", 2) == 0){
     		outPhysicalValue = FReal(-0.82);
     		outIndex = OW;
     	}
     	else if( strncmp(type, "HW", 2) == 0){
     		outPhysicalValue = FReal(-0.41);
     		outIndex = HW;
     	}
     	else if( strncmp(type, "Na", 2) == 0){
     		outPhysicalValue = FReal(1.0);
     		outIndex = Na;
     	}
     	else if( strncmp(type, "Cl", 2) == 0){
     		outPhysicalValue = FReal(-1.0);
     		outIndex = Cl;
     	}
     	else{

     		std::cerr << "Atom type not defined "<< type << std::endl;
     		exit(-1);
     	}
   //  	std::cout << "Atom type " << type << "  "<< outPhysicalValue << " " << outIndex <<std::endl;
     }

} ;
/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FDlpolyLoader
* Please read the license
* Particle has to extend {FExtendPhysicalValue,FExtendPosition}
*/
template <class FReal>
class FDlpolyAsciiLoader : public FDlpolyLoader<FReal> {
private:
    std::ifstream file;         //< The file to read
//    FPoint<FReal>        centerOfBox;          //< The center of box read from file
//    FReal         boxWidth;             //< the box width read from file
//    int           nbParticles;            //< the number of particles read from file
//    int           levcfg  ;               //< DL_POLY CONFIG file key. 0,1 or 2
//    FReal         energy  ;
//public:
//    enum Type{
//        OW,
//        HW,
//        Na,
//        Cl,
//        Undefined,
//    };
//
    /**
    * The constructor need the file name
    * @param filename the name of the file to open
    * you can test if file is successfuly open by calling hasNotFinished()
        Box SPC water from DLPOLY TEST17
         2         1       417  -591626.141968
     17.200000000000      0.000000000000      0.000000000000
      0.000000000000     17.200000000000      0.000000000000
      0.000000000000      0.000000000000     17.200000000000
    */
public:
    FDlpolyAsciiLoader(const char* const filename): file(filename,std::ifstream::in){
        // test if open
        if(this->file.is_open()){
            const int bufferSize = 512;
            char buffer[bufferSize];
            file.getline(buffer, bufferSize);


            int imcon ;
            //int tempi(0);
            FReal tempf(0);
            file >> FDlpolyLoader<FReal>::levcfg >> imcon >> FDlpolyLoader<FReal>::nbParticles >> FDlpolyLoader<FReal>::energy;
            // Periodic case
            if( imcon > 0 ) {
                FReal widthx, widthy, widthz;
                file >> widthx >> tempf >> tempf;
                file >> tempf >> widthy >> tempf;
                file >> tempf >> tempf >> widthz;

                this->boxWidth = widthx;
            }
            // Non periodic case
            else{
                file >> this->boxWidth;
            }
            this->centerOfBox.setPosition(0.0,0.0,0.0);
        }
        else {
            this->boxWidth    = 0;
            this->nbParticles = 0;
        }

        std::cout << "ASCII LOADER "<< this->nbParticles<< "     "<< this->boxWidth<<std::endl;
    }
    /**
    * Default destructor, simply close the file
    */
    virtual ~FDlpolyAsciiLoader(){
        file.close();
    }

    /**
      * To know if file is open and ready to read
      * @return true if loader can work
      */
    bool isOpen() const{
        return this->file.is_open() && !this->file.eof();
    }


    /**
      * Fill a particle
      * @warning to work with the loader, particles has to expose a setPosition method
      * @param the particle to fill
        OW               1
             5.447823189       -0.4124521286        -3.845403447
           7.64746800518      -1.34490700206      -2.81036521708
          -4406.48579000       6815.52906417       10340.2577024
      */
    void fillParticle(FPoint<FReal>* inPosition, FReal inForces[3], FReal* inPhysicalValue, int* inIndex){

    	FReal x, y, z, fx=0.0, fy=0.0, fz=0.0, vx=0.0, vy=0.0, vz=0.0;
    	int index;


    	this->file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    	std::string atomType, line;
    	this->file >> atomType;
//
    	file >> index;

    	std::getline(file, line); // needed to skip the end of the line in non periodic case
  //  	std::cout << "line: " << line << std::endl;
        if ( FDlpolyLoader<FReal>::levcfg == 0) {
    		file >> x >> y >> z;
        }else if ( FDlpolyLoader<FReal>::levcfg == 1) {
    		file >> x >> y >> z;
    		file >> vx >> vy >> vz;
    	}else {
    		file >> x >> y >> z;
    		file >> vx >> vy >> vz;
    		file >> fx >> fy >> fz;
    	}

    	inIndex[0] = index;

    	inPosition->setPosition( x, y ,z);
    	inForces[0] = fx;
    	inForces[1] = fy;
    	inForces[2] = fz;

        FDlpolyLoader<FReal>::getPhysicalValue(atomType, *inPhysicalValue,*inIndex  );

    }
};


template <class FReal>
class FDlpolyBinLoader : public FDlpolyLoader<FReal> {
protected:
    FILE* const file;         //< The file to read
//    FPoint<FReal> centerOfBox;    //< The center of box read from file
//    double boxWidth;             //< the box width read from file
//    FSize nbParticles;            //< the number of particles read from file
//    double energy;
    size_t removeWarning;

    template<class Type>
    Type readValue(){
        int sizeBefore, sizeAfter;
        Type value;
        removeWarning = fread(&sizeBefore, sizeof(int), 1, file);
        removeWarning = fread(&value, sizeof(Type), 1, file);
        removeWarning = fread(&sizeAfter, sizeof(int), 1, file);
        if( sizeBefore != sizeof(Type) ) printf("Error in loader Dlpoly Size before %d should be %lu\n", sizeBefore, sizeof(Type));
        if( sizeAfter != sizeof(Type) ) printf("Error in loader Dlpoly Size after %d should be %lu\n", sizeAfter, sizeof(Type));
        return value;
    }

    template<class Type>
    Type* readArray(Type array[], const int size){
        int sizeBefore, sizeAfter;
        removeWarning = fread(&sizeBefore, sizeof(int), 1, file);
        removeWarning = fread(array, sizeof(Type), size, file);
        removeWarning = fread(&sizeAfter, sizeof(int), 1, file);
        if( sizeBefore != int(sizeof(Type) * size) ) printf("Error in loader Dlpoly Size before %d should be %lu\n", sizeBefore, size*sizeof(Type));
        if( sizeAfter != int(sizeof(Type) * size) ) printf("Error in loader Dlpoly Size after %d should be %lu\n", sizeAfter, size*sizeof(Type));
        return array;
    }

public:
    /**
    * The constructor need the file name
    * @param filename the name of the file to open
    * you can test if file is successfuly open by calling hasNotFinished()
        energy box size nb particles
        [index charge x y z fx fy fz]
        int double double ...
    */
    FDlpolyBinLoader(const char* const filename): file(fopen(filename, "rb")) {
        // test if open
        if(this->file != nullptr){
            FDlpolyLoader<FReal>::energy = readValue<double>();
            double boxDim[3];
            FDlpolyLoader<FReal>::boxWidth = readArray<double>(boxDim,3)[0];
            FDlpolyLoader<FReal>::nbParticles = readValue<int>();

            FDlpolyLoader<FReal>::centerOfBox.setPosition(0.0,0.0,0.0);
        }
        else {
            FDlpolyLoader<FReal>::boxWidth    = 0;
            FDlpolyLoader<FReal>::nbParticles = 0;
        }
            std::cout << "nbParticles  "<< FDlpolyLoader<FReal>::nbParticles <<std::endl;
    }
    /**
    * Default destructor, simply close the file
    */
    virtual ~FDlpolyBinLoader(){
        fclose(file);
    }

    /**
      * To know if file is open and ready to read
      * @return true if loader can work
      */
    bool isOpen() const{
        return this->file != nullptr;
    }

//    /**
//      * To get the number of particles from this loader
//      * @param the number of particles the loader can fill
//      */
//    FSize getNumberOfParticles() const{
//        return FSize(this->nbParticles);
//    }
//
//    /**
//      * The center of the box from the simulation file opened by the loader
//      * @return box center
//      */
//    FPoint<FReal> getCenterOfBox() const{
//        return this->centerOfBox;
//    }
//
//    /**
//      * The box width from the simulation file opened by the loader
//      * @return box width
//      */
//    FReal getBoxWidth() const{
//      return static_cast<FReal>(this->boxWidth);
//    }
//
//    FReal getEnergy() const{
//      return static_cast<FReal>(this->energy);
//    }

    /**
      * Fill a particle
      * @warning to work with the loader, particles has to expose a setPosition method
      * @param the particle to fill
        [index charge x y z fx fy fz]
      */
    void fillParticle(FPoint<FReal>* inPosition, FReal inForces[3], FReal* inPhysicalValue, int* inIndex){
        double x, y, z, fx, fy, fz, charge;
        int index;

        int size;
        removeWarning = fread(&size, sizeof(int), 1, file);
       if(size != 60) printf("Error in loader Dlpoly Size %d should be %d\n", size, 60);

        removeWarning = fread(&index, sizeof(int), 1, file);
        removeWarning = fread(&charge, sizeof(double), 1, file);

        removeWarning = fread(&x, sizeof(double), 1, file);
        removeWarning = fread(&y, sizeof(double), 1, file);
        removeWarning = fread(&z, sizeof(double), 1, file);

        removeWarning = fread(&fx, sizeof(double), 1, file);
        removeWarning = fread(&fy, sizeof(double), 1, file);
        removeWarning = fread(&fz, sizeof(double), 1, file);

        removeWarning = fread(&size, sizeof(int), 1, file);
        if(size != 60) printf("Error in loader Dlpoly Size %d should be %d\n", size, 60);

        inPosition->setPosition( static_cast<FReal>(x), static_cast<FReal>(y) ,static_cast<FReal>(z));
        inForces[0] = static_cast<FReal>(fx);
	inForces[1] = static_cast<FReal>(fy);
	inForces[2] = static_cast<FReal>(fz);
        *inPhysicalValue = static_cast<FReal>(charge);
        *inIndex = index;
    }

};


#endif //FDlpolyLoader_HPP


