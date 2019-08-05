// See LICENCE file at project root
// author Berenger Bramas and Olivier Coulaud
//
#ifndef FFmaGenericLoader_HPP
#define FFmaGenericLoader_HPP

#include <ios>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>

#include "Utils/FGlobal.hpp"
#include "Utils/FAssert.hpp"
#include "FAbstractLoader.hpp"
#include "Utils/FPoint.hpp"

#include "Containers/FOctree.hpp"


/** \brief Particle class used in FMA loader and writer.
 *
 *
 * The pieces of data are : PosX, PosY, PosZ, physicalValue,
 * Potential, forceX, forceY, forceZ. The first 4 are mandatory.
 * Data is stored as FReal.
 *
 * See FFmaGenericLoader for information on the format.
 *
 * \tparam READ  number of items to read (UNUSED, see WRITE)
 * \tparam WRITE number of items to write (must be >= 4)
 */
template<class FReal, unsigned int READ, unsigned int WRITE>
class FmaRWParticle {
    static_assert(WRITE >= 4, "Cannot create FmaRWParticle with less than 4 as value for WRITE");

    /// Data stored
    FReal data[WRITE];
public:
    FmaRWParticle() = default;

    /// Get a FPoint<FReal> from the position
    FPoint<FReal> getPosition() const{
        return FPoint<FReal>(data[0],data[1],data[2]);
    }
    /// Set the position from a FPoint<FReal>
    void setPosition(FPoint<FReal> & inPoint){
        data[0] = inPoint.getX();
        data[1] = inPoint.getY();
        data[2] = inPoint.getZ();
    }

    /// Set the position from a three FReal values
    void setPosition(const FReal& inX, const  FReal& inY, const  FReal &inZ){
        data[0] = inX;
        data[1] = inY;
        data[2] = inZ;
    }


    /// Get a FReal from the physicalValue
    FReal getPhysicalValue() const{
        return data[3];
    }

    /// Get a ptr to be able to set the physicalValue
    FReal* setPhysicalValue() {
        return &data[3];
    }
    /// Set the physicalValue
    void  setPhysicalValue(const  FReal& Q) {
        data[3] = Q;
    }

    /// Get a FReal from the potential
    FReal getPotential() const{
        FAssertLF(WRITE>4,"Cannot access to Potential with WRITE<=4");
        return data[4];
    }

    /// Get a ptr to be able to set the potential
    FReal* setPotential() {
        FAssertLF(WRITE>4,"Cannot set Potential with WRITE<=4");
        return &data[4];
    }
    /// Set the potential
    void  setPotential(const FReal& P) {
        FAssertLF(WRITE>4,"Cannot set Potential with WRITE<=4");
        data[4] = P;
    }
    /// Get a ptr to read the forces
    FReal* getForces() {
        FAssertLF(WRITE>7,"Cannot access to forces[] with WRITE<=8");
        return &data[5];
    }

    /// Get a ptr to write the forces
    FReal* setForces() {
        FAssertLF(WRITE>7,"Cannot set Forces[] with WRITE<=7");
        return &data[5];
    }
    /// Set the forces from three values
    void setForces(const FReal& inFX, const FReal& inFY, const FReal &inFZ){
        FAssertLF(WRITE>7,"Cannot set Forces[] with WRITE<=7");
        data[5] = inFX;
        data[6] = inFY;
        data[7] = inFZ;
    }


    /// Get directly a ptr to the data
    FReal  * getPtrFirstData(){
        return data;
    }
    /// Same as above with const qualifier
    const  FReal * getPtrFirstData() const{
        return data;
    }

    /// Get READ
    unsigned int getReadDataNumber() const{
        return (READ);
    }

    /// Get WRITE
    unsigned int getWriteDataNumber() const{
        return WRITE;
    }

    /// Get size of Class Particle
    unsigned int getWriteDataSize() const {
        return sizeof(FmaRWParticle<FReal, READ,WRITE>);
    }

    /// Get Size of array (should be same as above...)
    unsigned int getClassSize() const {
        return WRITE*sizeof(FReal);
    }
};



/**\class FFmaGenericLoader
 * \warning This class only works in shared memory (doesn't work with MPI).
 *
 * \brief Reads an FMA formated particle file.
 *
 * The file may be in ascii or binary mode.  There are several overloads of the
 * fillParticle(FPoint<FReal>*, FReal*) member function to read data from a file. The
 * example below shows how to use the loader to read from a file.
 *
 *
 * \code
 * // Instanciate the loader with the particle file.
 * FFmaGenericLoader<FReal> loader("../Data/unitCubeXYZQ20k.fma"); // extension fma -> ascii format
 * // Retrieve the number of particles
 * FSize nbParticles = loader.getNumberOfParticles();
 *
 * // Create an array of particles, initialize to 0.
 * FmaRParticle * const particles = new FmaRParticle[nbParticles];
 * memset(particles, 0, sizeof(FmaRParticle) * nbParticles) ;
 *
 * // Read the file via the loader.
 * for(FSize idx = 0 ; idx < nbParticles ; ++idx){
 *     loader.fillParticle(particles[idx]);
 * }
 * \endcode
 * ----------------------------------------
 * FMA is a simple format to store particles in a file. It is organized as follow.
 *
 * \code
 *   DatatypeSize  Number_of_record_per_line
 *   NB_particles  half_Box_width  Center_X  Center_Y  Center_Z
 *   Particle_values
 * \endcode
 *
 * `DatatypeSize` can have one of two values:
 *  - 4, float ;
 *  - 8, double.
 *
 * `Number_of_records_per_line` gives the data count for each line of
 * the `Particle_values`. For example :
 *  - 4, the particle values are X Y Z Q;
 *  - 8, the particle values are X Y Z Q  P FX FY FZ<br>
 *
 */
template <class FReal>
class FFmaGenericLoader : public FAbstractLoader<FReal>  {
protected:
    std::fstream* file;       ///< the stream used to read the file
    bool          binaryFile; ///< if true the file to read is in binary mode
    FPoint<FReal> centerOfBox;///< The center of box (read from file)
    FReal         boxWidth;   ///< the box width (read from file)
    FSize         nbParticles;///< the number of particles (read from file)
    unsigned int  typeData[2];///< Size of the data to read, number of data on 1 line

private:
    FReal *       tmpVal;     ///< Temporary array to read data
    /// Count of other data pieces to read in a particle record after the 4 first ones.
    unsigned int  otherDataToRead;

    void open_file(const std::string filename, const bool binary) {
            if(binary) {
                this->file = new std::fstream (filename.c_str(),std::ifstream::in| std::ios::binary);
            }
            else {
                this->file = new std::fstream(filename.c_str(),std::ifstream::in) ;
            }
            // test if open
            if(! this->file->is_open()){
                std::cerr << "File "<< filename<<" not opened! Error: " << strerror(errno) <<std::endl;
                std::exit( EXIT_FAILURE);
            }
            std::cout << "Opened file "<< filename << std::endl;
    }

public:
    /**
     * This constructor opens a file using the given mode and reads its
     * header. The file will be kept opened until destruction of the object.
     *
     * - All information accessible in the header can be retreived after this call.
     * - To test if the file has successfully been opened, call hasNotFinished().
     *
     * @param filename the name of the file to open
     * @param binary   true if the file to open is in binary mode
     */
    FFmaGenericLoader(const std::string & filename,const bool binary ):
        file(nullptr), binaryFile(binary), centerOfBox(0.0,0.0,0.0), boxWidth(0.0),
        nbParticles(0), tmpVal(nullptr), otherDataToRead(0)
        {
            this->open_file(filename, binary);
            this->readHeader();
	}

    /**
     * This constructor opens a file and reads its header. The file will be
     * kept opened until destruction of the object.
     *
     * - The opening mode is guessed from the file extension : `.fma` will open
     * in ASCII mode, `.bfma` will open in binary mode.
     * - All information accessible in the header can be retreived after this call.
     * - To test if the file has successfully been opened, call hasNotFinished().
     *
     * @param filename the name of the file to open. Must end with `.fma` or `.bfma`.
     */
    FFmaGenericLoader(const std::string & filename) : file(nullptr),binaryFile(false),
                                                      centerOfBox(0.0,0.0,0.0),boxWidth(0.0),nbParticles(0),tmpVal(nullptr),otherDataToRead(0) {
        // open particle file
        if( filename.find(".bfma") != std::string::npos ) {
            binaryFile = true;
        } else if( filename.find(".fma")!=std::string::npos ) {
            binaryFile = false;
        } else  {
            std::cout << "FFmaGenericLoader: "
                      << "Only .fma or .bfma input file are allowed. Got "
                      << filename << "."
                      << std::endl;
            std::exit ( EXIT_FAILURE) ;
        }

        this->open_file(filename, binaryFile);
        this->readHeader();
    }

    /**
     * Default destructor, closes the file
     */
    virtual ~FFmaGenericLoader(){
        file->close();
        delete file;
        delete[] tmpVal;
    }

    /**
     * To know if file is open and ready to read
     * @return true if loader can work
     */
    bool isOpen() const{
        return this->file->is_open() && !this->file->eof();
    }

    /**
     * To get the number of particles from this loader
     */
    FSize getNumberOfParticles() const{
        return this->getParticleCount();
    }

    /**
     * The center of the box from the simulation file opened by the loader
     * @return box center
     */
    FPoint<FReal> getCenterOfBox() const{
        return this->getBoxCenter();
    }

    /**
     * \brief Get the distribution particle count
     * \return The distribution particle count
     */
    FSize getParticleCount() const {
        return this->nbParticles;
    }

    /**
     * \brief Get distribution center
     * \return A point representing the box center
     */
    FPoint<FReal> getBoxCenter() const{
        return this->centerOfBox;
    }

    /**
     * The box width from the simulation file opened by the loader
     * @return box width
     */
    FReal getBoxWidth() const{
        return this->boxWidth;
    }
    /**
     * The box width from the simulation file opened by the loader
     * @return the number of data per record (Particle)
     */
    unsigned int getNbRecordPerline(){
        return typeData[1]; }
    /**
     * To know if the data are in float or in double type
     * @return the type of the values float (4) or double (8)
     */
    unsigned int getDataType(){
        return typeData[0]; }

    /**
     * Fills a particle from the current position in the file.
     *
     * @param outParticlePositions the position of particle to fill (FPoint<FReal> class)
     * @param outPhysicalValue     the physical value of particle to fill (FReal)
     */
    void fillParticle(FPoint<FReal>*const outParticlePositions, FReal*const outPhysicalValue){
        if(binaryFile){
            file->read((char*)(outParticlePositions), sizeof(FReal)*3);
            file->read((char*)(outPhysicalValue), sizeof(FReal));
            if(otherDataToRead> 0){
                file->read((char*)(this->tmpVal), sizeof(FReal)*otherDataToRead);
            }
        } else {
            FReal x,y,z;
            (*this->file)  >> x >> y >> z >> (*outPhysicalValue);
            outParticlePositions->setPosition(x,y,z);

            if(otherDataToRead> 0){
                for (FSize 	i = 0 ; i <otherDataToRead; ++i){
                    (*this->file) >> x ;
                }
            }
        }
    }

    /**
     * Fill a particle set from the current position in the file.
     *
     * @param dataToRead   array of type FReal. It contains all the values of a
     * particles (for instance X,Y,Z,Q, ..)
     *
     * @param nbDataToRead number of value to read (I.e. size of the array)
     */
    void fillParticle(FReal* dataToRead, const unsigned int nbDataToRead){
        if(binaryFile){
            file->read((char*)(dataToRead), sizeof(FReal)*nbDataToRead);
            if(nbDataToRead< typeData[1]){
                file->read((char*)(this->tmpVal), sizeof(FReal)*(typeData[1]-nbDataToRead));
            }
        }
        else{

            for (unsigned int i = 0 ; i <nbDataToRead; ++i){
                (*this->file)  >>dataToRead[i];
            }
            if(nbDataToRead< typeData[1]){
                FReal x;
                for (unsigned int 	i = 0 ; i <typeData[1]-nbDataToRead; ++i){
                    (*this->file) >> x ;
                }
            }
        }
    }

    /**
     * Fills a particle form the current position in the file
     *
     * @tparam dataPart  the particle class. It must implement the members
     * getPtrFirstData(), getReadDataNumber(). See FmaRWParticle.
     *
     * @param dataToRead the particle to fill.
     */
    template <class dataPart>
    void fillParticle(dataPart& dataToRead){
        FSize otherDataRead = typeData[1] - dataToRead.getReadDataNumber() ;
        if(binaryFile){
            file->read((char*)(dataToRead.getPtrFirstData()), sizeof(FReal)*(dataToRead.getReadDataNumber()));
            if( otherDataRead > 0){
                file->read((char*)(this->tmpVal), sizeof(FReal)*(otherDataRead));
            }
        }
        else{
            FReal * val = dataToRead.getPtrFirstData();
            for (FSize i = 0 ; i <dataToRead.getReadDataNumber(); ++i){
                (*this->file)  >>*val;
                ++val;
            }
            if( otherDataRead > 0){
                FReal x;
                for (FSize i = 0 ; i <otherDataRead ;++i){
                    (*this->file)  >>x;
                }
            }
        }
    }

    /**
     * Fill a set of particles form the current position in the file.
     *
     * If the file is a binary file and we read all record per particle then we
     * read and fill the array in one instruction.
     *
     * @tparam dataPart  the particle class. It must implement the members
     * getPtrFirstData(), getReadDataNumber(). See FmaRWParticle.
     *
     * @param dataToRead the array of particles to fill.
     * @param N          the number of particles.
     */
    template <class dataPart>
    void fillParticle(dataPart *dataToRead, const FSize N){
        FSize otherDataRead = typeData[1] - (*dataToRead).getReadDataNumber() ;
        if (otherDataRead < 0){
            std::cerr << "Error in fFFmaGenericLoader::fillParticle(dataPart *dataToRead, const FSize N)."
                      << std::endl
                      << "Wrong number of values to read:" << std::endl
                      << "expected " << typeData[1] << " from file\n"
                      << "expected " << (*dataToRead).getReadDataNumber() << "from structure."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(binaryFile && otherDataRead == 0 ){
            file->read((char*)((*dataToRead).getPtrFirstData()),
                       sizeof(FReal)*(N*(*dataToRead).getReadDataNumber()));
        }
        else {
            for (FSize i = 0 ; i <N; ++i) {
                this->fillParticle(dataToRead[i]) ;
            }
        }
    }

private:
    void readHeader() {
        if(this->binaryFile){
            this->readBinaryHeader();
        }
        else {
            this->readAscciHeader();
        }

        std::cout << "   nbParticles: " <<this->nbParticles << std::endl
                  << "   Box width:   " <<this->boxWidth << std::endl
                  << "   Center:        " << this->centerOfBox << std::endl;
    }
    void readAscciHeader() {
        std::cout << " File open in ASCII mode "<< std::endl ;
        FReal x,y,z;
        (*this->file) >> typeData[0]>> typeData[1];
        std::cout << "   Datatype "<< typeData[0] << " "<< typeData[1] << std::endl;
        (*this->file) >> this->nbParticles >> this->boxWidth >> x >> y >> z;
        this->centerOfBox.setPosition(x,y,z);
        this->boxWidth *= 2;
        otherDataToRead = typeData[1] -  (unsigned int)(4);
    };
    void readBinaryHeader(){
        std::cout << " File open in binary mode "<< std::endl;
        file->seekg (std::ios::beg);
        file->read((char*)&typeData,2*sizeof(unsigned int));
        std::cout << "   Datatype "<< typeData[0] << " "<< typeData[1] << std::endl;
        if(typeData[0] != sizeof(FReal)){
            std::cerr << "Size of elements in part file " << typeData[0] << " is different from size of FReal " << sizeof(FReal)<<std::endl;
            std::exit( EXIT_FAILURE);
        }
        else{
            file->read( (char*)&(this->nbParticles), sizeof(FSize) );
            file->read( (char*)&(this->boxWidth) ,sizeof(this->boxWidth) );
            this->boxWidth *= 2;

            FReal x[3];
            file->read( (char*)x,sizeof(FReal)*3);
            this->centerOfBox.setPosition(x[0],x[1],x[2]);
        }
        otherDataToRead = typeData[1] - (unsigned int)(4);
        if(otherDataToRead>0){
            tmpVal = new FReal[otherDataToRead];
        }
    }

};


/**
 * \warning This class only works in shared memory (doesn't work with MPI).
 *
 * \brief Writes a set of particles to an FMA formated file.
 *
 * The file may be in ASCII or binary mode. The example below shows how to use the class.
 *
 * \code
 * // Instanciate the writer with a binary fma file (extension .bfma).
 * FFmaGenericWriter<FReal> writer ("data.bfma");
 *
 * // Write the header of the file.
 * writer.writeHeader(loader.getCenterOfBox(), loader.getBoxWidth(), NbPoints, sizeof(FReal), nbData);
 *
 * // Write the data. Here particles is an array and a particle has nbData values.
 * writer.writeArrayOfReal(particles, nbData, NbPoints);
 * \endcode
 * ----------------------------------------
 * FMA is a simple format to store particles in a file. It is organized as follow.
 *
 * \code
 *   DatatypeSize  Number_of_record_per_line
 *   NB_particles  half_Box_width  Center_X  Center_Y  Center_Z
 *   Particle_values
 * \endcode
 *
 * `DatatypeSize` can have one of two values:
 *  - 4, float;
 *  - 8, double.
 *
 * `Number_of_records_per_line` gives the data count for each line of
 * the `Particle_values`. For example :
 *  - 4, the particle values are `X Y Z Q`;
 *  - 8, the particle values are `X Y Z Q  P FX FY FZ`.
 */
template <class FReal>
class FFmaGenericWriter {

protected:
    std::fstream *file;  ///< the stream used to read the file
    bool binaryFile  ;   ///< if true the file to read is in binary mode

public:
    /**
     * This constructor opens a file to be written to.
     *
     * - The opening mode is guessed from the file extension : `.fma` will open
     * in ASCII mode, `.bfma` will open in binary mode.
     *
     * @param filename the name of the file to open.
     */
    FFmaGenericWriter(const std::string & filename): binaryFile(false) {
        std::string ext(".bfma");
        // open particle file
        if(filename.find(".bfma") !=std::string::npos) {
            binaryFile = true;
            this->file = new std::fstream (filename.c_str(),std::ifstream::out| std::ios::binary);
        }
        else if(filename.find(".fma")!=std::string::npos ) {
            this->file = new std::fstream(filename.c_str(),std::ifstream::out) ;
            this->file->precision(10);
        }
        else  {
            std::cout << "Input file not allowed only .fma or .bfma extensions" <<std::endl;
            std::exit ( EXIT_FAILURE) ;
        }
        // test if open
        if(! this->file->is_open()){
            std::cerr << "File " << filename << " not opened! " << std::endl;
            std::exit( EXIT_FAILURE);
        }
    }

    /**
     * This constructor opens a file to be written to.
     *
     * @param filename the name of the file to open.
     * @param binary   true if the file to open is in binary mode
     */
    FFmaGenericWriter(const std::string & filename, const bool binary ) : file(nullptr), binaryFile(binary)
	{
            if(binary) {
                this->file = new std::fstream (filename.c_str(),std::ifstream::out| std::ios::binary);
            }
            else {
                this->file = new std::fstream(filename.c_str(),std::ifstream::out) ;
                this->file->precision(10);
            }
            // test if open
            if(! this->file->is_open()){
                std::cerr << "File "<< filename<<" not opened! " <<std::endl;
                std::exit( EXIT_FAILURE);
            }
	}

    /**
     * Default destructor, closes the file.
     */
    virtual ~FFmaGenericWriter(){
        file->close();
        delete file ;
    }

    /**
     * To know if file is open and ready to read
     * @return true if loader can work
     */
    bool isOpen() const{
        return this->file->is_open() && !this->file->eof();
    }

    /**
     * Writes the header of the FMA file
     * \warning All values inside typePart should be of the same type (float or double)
     *
     * @param centerOfBox  The centre of the Box (FPoint<FReal> class)
     * @param boxWidth     The width of the box
     * @param nbParticles  Number of particles in the box (or to save)
     * @param data         Data type of the particle class (FmaBasicParticle, FmaRParticle or FmaParticle)
     */
    template <class typePart>
    void writeHeader(const FPoint<FReal> &centerOfBox,const FReal &boxWidth, const FSize &nbParticles, const typePart  data) {
        unsigned int typeFReal[2]  = {sizeof(FReal) , sizeof(typePart) / sizeof(FReal) };
        const unsigned int ndata  = data.getWriteDataNumber();
        std::cout <<"    WriteHeader: typeFReal: " << typeFReal[0]  << "  nb Elts: " << typeFReal[1]  <<"   NData to write "<< ndata<< "\n";
        if (ndata != typeFReal[1]){
            typeFReal[1] = ndata;
        }
        FReal x = boxWidth * FReal(0.5);
        if(this->binaryFile) {
            this->writerBinaryHeader(centerOfBox,x,nbParticles,typeFReal);
        }
        else {
            this->writerAscciHeader(centerOfBox,x,nbParticles,typeFReal);
        }
    }

    /**
     * Writes the header of FMA file.
     *
     * Should be used if we write the particles with writeArrayOfReal method
     *
     * @param centerOfBox      The center of the Box (FPoint<FReal> class)
     * @param boxWidth         The width of the box
     * @param nbParticles      Number of particles in the box (or to save)
     * @param dataType         Size of the data type of the values in particle
     * @param nbDataPerRecord  Number of record/value per particle
     */
    void writeHeader(const FPoint<FReal> &centerOfBox,const FReal &boxWidth, const FSize &nbParticles,
                     const unsigned int  dataType, const unsigned int  nbDataPerRecord) {
        unsigned int typeFReal[2]  = {dataType , nbDataPerRecord };
        FReal x = boxWidth * FReal(0.5);
        if(this->binaryFile) {
            this->writerBinaryHeader(centerOfBox,x,nbParticles,typeFReal);
        }
        else {
            this->writerAscciHeader(centerOfBox,x,nbParticles,typeFReal);
        }
    }

    /**
     * Writes an array of particles to the file.
     *
     * @warning dataPart should be FmaBasicParticle, FmaRParticle or FmaParticle
     * or should implement all methods inside FmaRWParticle.
     *
     * @tparam dataPart   The class of the particle array.
     * @param dataToWrite Array of particles of type dataPart
     * @param N           Number of element in the array
     *
     * Example 1
     * \code
     * FmaRParticle *  particles = new FmaRParticle[nbParticles];
     * memset(particles, 0, sizeof(FmaRParticle) * nbParticles) ;
     * ...
     * FFmaGenericWriter<FReal> writer(filenameOut) ;
     * Fwriter.writeHeader(Centre,BoxWith, nbParticles,*particles) ;
     * Fwriter.writeArrayOfParticles(particles, nbParticles);
     * \endcode
     *
     * Example2
     * \code
     * FReal *  particles = new FReal[4*NbPoints] ; // store 4 data per particle
     * memset(particles, 0, sizeof(FmaRParticle) * nbParticles) ;
     * ...
     * FmaBasicParticle *ppart = (FmaBasicParticle*)(&particles[0]);
     * FFmaGenericWriter<FReal> writer(filenameOut) ;
     * writer.writeHeader(Centre,BoxWith, nbParticles,*particles) ;
     * writer.writeArrayOfParticles(particles, nbParticles);
     *  \endcode
     */
    template <class dataPart>
    void writeArrayOfParticles(const dataPart *dataToWrite, const FSize N){
        //		std::cout << "NB points to write: "<< N <<std::endl;
        if(binaryFile){
            unsigned int recordSize=  dataToWrite[0].getWriteDataSize() ;
            unsigned int typeFReal[2]      = {sizeof(FReal) , sizeof(dataPart) / sizeof(FReal) };
            // std::cout << "typeData "<< typeFReal[0] << " "<< typeFReal[1] <<"  "<< std::endl;

            if (sizeof(dataPart) == recordSize){
                // std::cout << "Size to write:  "<<N*dataToWrite[0].getWriteDataSize() <<std::endl;
                file->write((const char*)(dataToWrite[0].getPtrFirstData()), N*recordSize);
            } else {
                file->write((const char* )&typeFReal[0],2*sizeof(FSize));
                // std::cout << "Size to write:   N* "<<typeFReal[0] *typeFReal[1]  <<std::endl;
                for (FSize i = 0 ; i <N ; ++i){
                    file->write((const char*)(dataToWrite[i].getPtrFirstData()), recordSize );
                    // const FReal * val = dataToWrite[i].getPtrFirstData() ;
                    // std::cout << i <<"   ";
                    // for( FSize j=0; j<typeFReal[1] ; ++j){
                    // 	std::cout << *val << "   ";++val;
                    // }
                    // std::cout <<std::endl;
                }
            }
        }
        else{ // ASCII part
            const unsigned int ndata = dataToWrite[0].getWriteDataNumber();
            // std::cout << "typeData "<< sizeof(FReal) << " "<<ndata << std::endl;
            this->file->precision(10);

            for (FSize i = 0 ; i <N ; ++i){
                const FReal * val = dataToWrite[i].getPtrFirstData() ;
                for (unsigned int j= 0 ; j <ndata ; ++j){
                    (*this->file)  << *val << "    "; ++val;
                }
                (*this->file)  <<std::endl;
            }
        }
    }

    /**
     *  Write an array of data in a file Fill
     *
     * @param dataToWrite array of particles of type FReal
     * @param nbData number of data per particle
     * @param N number of particles
     *
     *   The size of the array is N*nbData
     *
     *   example
     * \code
     * FmaRParticle * const particles = new FmaRParticle[nbParticles];
     * memset(particles, 0, sizeof(FmaRParticle) * nbParticles) ;
     * ...
     * FFmaGenericWriter<FReal> writer(filenameOut) ;
     * Fwriter.writeHeader(Centre,BoxWith, nbParticles,*particles) ;
     * Fwriter.writeArrayOfReal(particles, nbParticles);
     * \endcode
     */
    void writeArrayOfReal(const FReal *dataToWrite, const FSize nbData, const FSize N){
        if(binaryFile){
            file->write((const char*)(dataToWrite), N*nbData*sizeof(FReal));
        }
        else{
            this->file->precision(10);
            //			std::cout << "N "<< N << " nbData "<< nbData<<std::endl;
            //			exit(-1);
            FSize k = 0;
            for (FSize i = 0 ; i <N ; ++i){
                //				std::cout << "i "<< i << "  ";
                for (unsigned int jj= 0 ; jj<nbData ; ++jj, ++k){
                    (*this->file)  << dataToWrite[k] << "    ";
                    //					std::cout      << dataToWrite[k]<< "  ";
                }
                (*this->file)  <<std::endl;
                //				std::cout <<std::endl;
            }
            //			std::cout << "END"<<std::endl;
        }
    }

    /**
     *  Write all particles inside the octree
     *
     * @param tree Octree that contains the particles in the leaves
     * @param N number of particles
     *
     *   example
     * \code
     *  OctreeClass tree(TreeHeight, SubTreeHeight, BoxWidth, CenterOfBox);
     * ...
     * FFmaGenericWriter<FReal> writer(filenameOut) ;
     * Fwriter.writeHeader(Centre,BoxWith, nbParticles,*particles) ;
     * Fwriter.writeDistributionOfParticlesFromOctree(&tree, nbParticles);
     * \endcode
     */
    template <class Toctree>
    void writeDistributionOfParticlesFromOctree( Toctree *tree, const FSize N){
        //
	this->writeHeader(tree->getBoxCenter(),tree->getBoxWidth(), N, sizeof(FReal), 4 );

        tree->forEachLeaf([this]( typename Toctree::LeafClassType* leaf){
                FReal * particles = nullptr;
                FSize nbMaxParticlesInLeaf = 0 ;
                //
                // Input
                const FReal*const posX = leaf->getTargets()->getPositions()[0];
                const FReal*const posY = leaf->getTargets()->getPositions()[1];
                const FReal*const posZ = leaf->getTargets()->getPositions()[2];
                const FReal*const physicalValues = leaf->getTargets()->getPhysicalValues();
                //
                const FSize nbParticlesInLeaf = leaf->getTargets()->getNbParticles();
                if (nbParticlesInLeaf > nbMaxParticlesInLeaf ) {
                    nbMaxParticlesInLeaf	 = nbParticlesInLeaf ;
                    particles = new FReal[4*nbMaxParticlesInLeaf] ;
                }
                //
                for(FSize idxPart = 0 , j = 0; idxPart < nbParticlesInLeaf ; ++idxPart, j+=4){
                    particles[j]      = posX[idxPart] ;
                    particles[j+1]  = posY[idxPart] ;
                    particles[j+2]  = posZ[idxPart] ;
                    particles[j+3]  = physicalValues[idxPart] ;
                }
                this->writeArrayOfReal(particles, 4, nbParticlesInLeaf) ;

                delete [] particles ;
            });
    }


private:
    void writerAscciHeader( const FPoint<FReal> &centerOfBox,const FReal &boxWidth,
                            const FSize &nbParticles, const unsigned int *typeFReal) {
        this->file->precision(10);
        (*this->file) << typeFReal[0] <<"   "<<typeFReal[1]<<std::endl;
        (*this->file) << nbParticles << "   "<<  boxWidth << "   "
                      <<  centerOfBox.getX()  << "  " << centerOfBox.getY() << " "<<centerOfBox.getZ()
                      << std::endl;
    }
    void writerBinaryHeader(const FPoint<FReal> &centerOfBox,const FReal &boxWidth,
                            const FSize &nbParticles, const unsigned int *typeFReal) {
        file->seekg (std::ios::beg);
        file->write((const char*)typeFReal,2*sizeof(unsigned int));
        if(typeFReal[0]  != sizeof(FReal)){
            std::cout << "Size of elements in part file " << typeFReal[0] << " is different from size of FReal " << sizeof(FReal)<<std::endl;
            std::exit( EXIT_FAILURE);
        }
        else{
            file->write( (const char*)&(nbParticles), sizeof(FSize) );
            //			std::cout << "nbParticles "<< nbParticles<<std::endl;
            file->write( (const char*)&(boxWidth) ,sizeof(boxWidth) );
            file->write( (const char*)(centerOfBox.getDataValue()),sizeof(FReal)*3);
        }
    }
};



#endif //FFmaGenericLoader_HPP
