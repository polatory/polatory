// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tools/fma_loader.hpp
// --------------------------------
#ifndef SCALFMM_TOOLS_FMA_LOADER_HPP
#define SCALFMM_TOOLS_FMA_LOADER_HPP

#include "scalfmm/container/particle.hpp"
#include "scalfmm/container/particle_container.hpp"
#include "scalfmm/container/point.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/for_each.hpp"

#include <array>
#include <cstdlib>
#include <errno.h>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace scalfmm::io
{
    /**
     * @author Olivier Coulaud (olivier coulaud@inria.fr)
     * @warning This class only works in shared memory (doesn't work with MPI).
     *
     * @brief Reads an FMA formatted particle file.
     *
     * The file may be in ascii or binary mode.  There are several overloads of the
     * fillParticle(FPoint<FReal>*, FReal*) member function to read data from a file. The
     * example below shows how to use the loader to read from a file.
     *
     * @code
     * // Instanciate the loader with the particle file.
     * FFmaGenericLoader<FReal> loader("../Data/unitCubeXYZQ20k.fma"); // extension fma -> ascii format
     * // Retrieve the number of particles
     * std::size_t nbParticles = loader.getNumberOfParticles();
     *
     * // Create an array of particles, initialize to 0.
     * FmaRParticle * const particles = new FmaRParticle[nbParticles];
     * memset(particles, 0, sizeof(FmaRParticle) * nbParticles) ;
     *
     * // Read the file via the loader.
     * for(std::size_t idx = 0 ; idx < nbParticles ; ++idx){
     *     loader.fillParticle(particles[idx]);
     * }
     * @endcode
     * ----------------------------------------
     * FMA is a simple format to store particles in a file. It is organized as follow.
     *
     * @code
     *   DatatypeSize  Number_of_record_per_line dimension Number_of_input_data
     *   NB_particles  half_Box_width  Center (dim values)
     *   Particle_values
     * @endcode
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
     * @tparam FReal
     * @tparam Dimension
     */
    template<class FReal, int Dimension = 3>
    class FFmaGenericLoader
    {
      protected:
        std::fstream* m_file;                               ///< the stream used to read the file
        bool m_binaryFile;                                  ///< if true the file to read is in binary mode
        container::point<FReal, Dimension> m_centerOfBox;   ///< The center of box (read from file)
        std::vector<FReal> m_center{};                      ///< The center of box (read from file)
        FReal m_boxWidth;                                   ///< the box width (read from file)
        std::size_t m_nbParticles;                          ///< the number of particles (read from file)
        std::array<unsigned int, 4> m_typeData;   ///< Size of the data to read, number of data on 1 line, dimension
                                                  ///< of space and number of input values
        std::string m_filename;                   ///< file name containung the data
        bool m_verbose;                           ///<  Verbose mode

      private:
        /**
         * @brief Temporary array to read data.
         *
         */
        FReal* m_tmpVal;

        /**
         * @brief Count of other data pieces to read in a particle record after the 4 first ones.
         *
         */
        unsigned int m_otherDataToRead;

        /**
         * @brief
         *
         * @param filename
         * @param binary
         * @param verbose
         */
        auto open_file(const std::string filename, const bool binary, const bool verbose = true) -> void
        {
            m_filename = filename;
            m_verbose = verbose;
            if(binary)
            {
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::in | std::ios::binary);
            }
            else
            {
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::in);
            }
            // test if open
            if(!this->m_file->is_open())
            {
                std::cerr << "File " << filename << " not opened! Error: " << strerror(errno) << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if(m_verbose)
            {
                std::cout << "FFmaGenericLoader file " << filename << " opened" << std::endl;
            }
        }

      public:
        using dataType = FReal;

        /**
         * @brief Construct a new FFmaGenericLoader object
         *
         * This constructor opens a file and reads its header. The file will be
         * kept opened until destruction of the object.
         *
         * - The opening mode is guessed from the file extension : `.fma` will open
         * in ASCII mode, `.bfma` will open in binary mode.
         * - All information accessible in the header can be retreived after this call.
         * - To test if the file has successfully been opened, call hasNotFinished().
         *
         * @param filename the name of the file to open. Must end with `.fma` or `.bfma`.
         * @param verbose
         * @param old_format
         */
        FFmaGenericLoader(const std::string& filename, bool verbose, bool old_format = false)
          : m_file(nullptr)
          , m_binaryFile(false)
          , m_centerOfBox{0.0}
          , m_boxWidth(0.0)
          , m_nbParticles(0)
          , m_filename(filename)
          , m_verbose(verbose)
          , m_tmpVal(nullptr)
          , m_otherDataToRead(0)
        {
            // open particle file
            if(filename.find(".bfma") != std::string::npos)
            {
                m_binaryFile = true;
            }
            else if(filename.find(".fma") != std::string::npos)
            {
                m_binaryFile = false;
            }
            else
            {
                std::cout << "FFmaGenericLoader: "
                          << "Only .fma or .bfma input file are allowed. Got " << filename << "." << std::endl;
                std::exit(EXIT_FAILURE);
            }

            this->open_file(filename, m_binaryFile, m_verbose);
            this->readHeader(old_format);
        }

        /**
         * @brief
         *
         */
        inline auto close() -> void
        {
            m_file->close();
            delete m_file;
            m_file = nullptr;
        }

        /**
         * @brief Destroy the FFmaGenericLoader object and close the file.
         *
         */
        virtual ~FFmaGenericLoader()
        {
            if(m_file != nullptr)
            {
                m_file->close();
            }
            delete[] m_tmpVal;
        }

        /**
         * @brief To know if the file is open and ready to read.
         * @return true if loader can work
         */
        bool isOpen() const { return this->m_file->is_open() && !this->m_file->eof(); }

        /**
         * @brief To get the number of particles from this loader
         */
        std::size_t getNumberOfParticles() const { return this->getParticleCount(); }
        /**
         *  @brief To get the number of particles from this loader
         */
        std::size_t getMyNumberOfParticles() const { return this->getParticleCount(); }

        /**
         * @brief To get the center of the box from the simulation file opened by the loader.
         *
         *  @return box center (type Point)
         */
        inline auto getCenterOfBox() const -> container::point<FReal, Dimension> { return this->getBoxCenter(); }
        /**
         * @brief Get the center of the box contining the particles
         *
         * @return A point (ontainer::point<FReal>) representing the box center
         */
        inline auto getBoxCenter() const { return this->m_centerOfBox; }
        /**
         * @brief Returns a pointer on the element of the Box center.
         *
         * @return FReal*
         */
        inline auto getPointerCenterOfBox() const -> FReal* { return this->m_center.data(); }

        /**
         * @brief Get the distribution particle count
         *
         * @return The distribution particle count
         */
        inline auto getParticleCount() const -> std::size_t { return this->m_nbParticles; }

        /**
         * @brief box width from the simulation file opened by the loader
         *
         * @return box width
         */
        inline auto getBoxWidth() const -> FReal { return this->m_boxWidth; }

        /**
         * @brief The box width from the simulation file opened by the loader
         *
         * @return the number of data per record (Particle)
         */
        inline auto getNbRecordPerline() -> unsigned int { return m_typeData[1]; }

        /**
         * @brief The Dimension space of the simulation file opened by the loader
         *
         * @return the Dimension space
         */
        inline auto get_dimension() -> unsigned int { return m_typeData[2]; }

        /**
         * @brief The box width from the simulation file opened by the loader
         *
         * @return the number of input data per record (Particle)
         */
        inline auto get_number_of_input_per_record() -> unsigned int { return m_typeData[3]; }

        /**
         * @brief The box width from the simulation file opened by the loader
         *
         * @return the number of ioutput data per record (Particle)
         */
        inline auto get_number_of_output_per_record() -> unsigned int
        {
            return m_typeData[1] - m_typeData[2] - m_typeData[3];
        }

        /**
         * @brief To know if the data are in float or in double type
         *
         * @return the type of the values float (4) or double (8)
         */
        inline auto getDataType() -> unsigned int { return m_typeData[0]; }

        /**
         * @brief Get the header size object
         *
         * @return auto
         */
        auto get_header_size()
        {
            return m_typeData.size() * sizeof(unsigned int) + sizeof(std::size_t) + sizeof(m_boxWidth) +
                   sizeof(FReal) * m_typeData[2];
        }

        /**
         * @brief Fill a particle set from the current position in the file.
         *
         * @param dataToRead   array of type FReal. It contains all the values of a
         * particles (for instance X,Y,Z,Q, ..)
         * @param nbDataToRead number of value to read (I.e. size of the array)
         */
        auto fillParticle(FReal* dataToRead, const unsigned int nbDataToRead) -> void
        {
            if(m_binaryFile)
            {
                m_file->read((char*)(dataToRead), sizeof(FReal) * nbDataToRead);
                if(nbDataToRead < m_typeData[1])
                {
                    m_file->read((char*)(this->m_tmpVal), sizeof(FReal) * (m_typeData[1] - nbDataToRead));
                }
            }
            else
            {
                // std::cout << " read " << nbDataToRead << " of " << m_typeData[1] << "  " << std::flush;
                for(unsigned int i = 0; i < nbDataToRead; ++i)
                {
                    (*this->m_file) >> dataToRead[i];
                    // std::cout << dataToRead[i] << " " << std::flush;
                    ;
                }
                // std::cout << '\n';
                if(nbDataToRead < m_typeData[1])   // skip extra data
                {
                    // std::cout << "extra value \n";

                    FReal x;
                    for(unsigned int i = 0; i < m_typeData[1] - nbDataToRead; ++i)
                    {
                        (*this->m_file) >> x;
                    }
                }
            }
        }

        /**
         * @brief Fills a set of particles form the current position in the file.
         *
         * If the file is a binary file and we read all record per particle then we
         * read and fill the array in one instruction.
         *
         * @tparam VectorType
         * @param dataToRead the array of particles to fill.
         * @param N          the number of particles.
         */
        template<class VectorType>
        auto fillParticles(VectorType& dataToRead, const std::size_t N) -> void
        {
            if(dataToRead.size() != m_typeData[1] * N)
            {
                std::cerr << "Error in fFFmaGenericLoader::fillParticle(dataPart *dataToRead, const  std::size_t N)."
                          << std::endl
                          << "Wrong number of values to read:" << std::endl
                          << "expected " << m_typeData[1] << " from file\n"
                          << "expected " << dataToRead.size() / N << " from structure." << std::endl;
                std::cerr << "Read from file: " << this->m_filename << std::endl;
                throw(" ToDo\n ");
                std::exit(EXIT_FAILURE);
            }

            if(m_binaryFile)
            {
                m_file->read((char*)(dataToRead.data()), sizeof(FReal) * m_typeData[1] * N);
            }
            else
            {
                for(std::size_t i = 0; i < dataToRead.size(); i += m_typeData[1])
                {
                    this->fillParticle(&(dataToRead[i]), m_typeData[1]);
                }
            }
        }

      private:
        /**
         * @brief
         *
         * @param old_format
         */
        auto readHeader(const bool old_format = true) -> void
        {
            int nbval = 4;
            if(old_format)
            {
                nbval = 2;
            }
            if(this->m_binaryFile)
            {
                this->readBinaryHeader(nbval);
            }
            else
            {
                this->readAscciHeader(nbval);
            }

            if(m_verbose)
            {
                std::cout << "   nbParticles: " << this->m_nbParticles << std::endl
                          << "   Box width:   " << this->m_boxWidth << std::endl
                          << "   Center:       [ ";
                for(unsigned int i = 0; i < m_typeData[2] - 1; ++i)
                {
                    std::cout << this->m_centerOfBox[i] << ", ";
                }
                std::cout << this->m_centerOfBox[m_typeData[2] - 1] << " ]" << std::endl;
            }
        }

        /**
         * @brief
         *
         * @param nbVal
         */
        auto readAscciHeader(const int nbVal) -> void
        {
            if(m_verbose)
            {
                std::cout << " File open in ASCII mode " << nbVal << std::endl;
                // std::cout << "   Datatype ";
            }
            FReal x;
            for(int i = 0; i < nbVal; ++i)
            {
                (*this->m_file) >> m_typeData[i];
                // std::cout << i << "  " << m_typeData[i] << std::endl;
            }

            if(nbVal == 2)   // Old Format
            {
                m_typeData[2] = 3;
                m_typeData[3] = 1;
            }
            if(Dimension != m_typeData[2])
            {
                std::cerr << "Wrong dimension. Template parameter is " << Dimension << " and dimension in file is "
                          << m_typeData[2] << ".\n";
                throw std::invalid_argument("Use argument -d to set the right dimension !");
            }
            (*this->m_file) >> this->m_nbParticles >> this->m_boxWidth;
            // if(m_verbose)
            // {
            //     std::cout << this->m_nbParticles << "  " << this->m_boxWidth << '\n';
            // }
            m_center.resize(m_typeData[2]);
            for(unsigned int i = 0; i < m_typeData[2]; ++i)
            {
                (*this->m_file) >> x;
                this->m_centerOfBox[i] = x;
                this->m_center[i] = x;
            }
            this->m_boxWidth *= 2;

            m_otherDataToRead = m_typeData[1] - m_typeData[2] - m_typeData[3];   // output variables
        }

        /**
         * @brief
         *
         * @param nbVal
         */
        auto readBinaryHeader(const int nbVal) -> void
        {
            if(m_verbose)
            {
                std::cout << " File open in binary mode " << std::endl;
            }
            m_file->seekg(std::ios::beg);
            m_file->read((char*)&m_typeData, nbVal * sizeof(unsigned int));
            if(nbVal == 2)   // Old Format
            {
                m_typeData[2] = 3;
                m_typeData[3] = 1;
            }
            if(Dimension != m_typeData[2])
            {
                std::cerr << "Wrong dimension. Template parameter is " << Dimension << " and dimension in file is "
                          << m_typeData[2] << ".\n";
                throw std::invalid_argument("Use argument -d to set the right dimension !");
            }
            if(m_typeData[0] != sizeof(FReal))
            {
                std::cerr << "readBinaryHeader: Size of elements in part file " << m_typeData[0]
                          << " is different from size of FReal " << sizeof(FReal) << std::endl;
                std::cerr << "The conversion is not yet implemented for binary file\n";
                std::exit(EXIT_FAILURE);
            }
            else
            {
                m_file->read((char*)&(this->m_nbParticles), sizeof(std::size_t));
                m_file->read((char*)&(this->m_boxWidth), sizeof(this->m_boxWidth));
                this->m_boxWidth *= 2;

                FReal* x = new FReal[m_typeData[2]];
                m_file->read((char*)x, sizeof(FReal) * m_typeData[2]);
                if(m_typeData[2] != Dimension)
                {
                    std::cerr << "m_typeData[2] != Dimension: " << m_typeData[2] << " != " << Dimension << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                // std::cout << " " << m_typeData[2] << "  centre: ";

                for(unsigned int i = 0; i < m_typeData[2]; ++i)
                {
                    this->m_centerOfBox[i] = x[i];
                }
                // std::cout << this->m_centerOfBox << std::endl;
            }
            m_otherDataToRead = m_typeData[1] - m_typeData[2] - m_typeData[3];   // output variables
            if(m_otherDataToRead > 0)
            {
                m_tmpVal = new FReal[m_otherDataToRead];
            }
        }
    };

    /**
     * @warning This class only works in shared memory (doesn't work with MPI).
     *
     * @brief Writes a set of particles to an FMA formated file.
     *
     * The file may be in ASCII or binary mode. The example below shows how to use the class.
     *
     * @code
     * // Instantiate the writer with a binary fma file (extension .bfma).
     * FFmaGenericWriter<FReal> writer ("data.bfma");
     *
     * // Write the header of the file.
     * writer.writeHeader(loader.getCenterOfBox(), loader.getBoxWidth(), NbPoints, sizeof(FReal), nbData);
     *
     * // Write the data. Here particles is an array and a particle has nbData values.
     * writer.writeArrayOfReal(particles, nbData, NbPoints);
     * @endcode
     * ----------------------------------------
     * FMA is a simple format to store particles in a file. It is organized as follow.
     *
     * @code
     *   DatatypeSize  Number_of_record_per_line dimension Number_of_input_data
     *   NB_particles  half_Box_width  Center (dim values)
     *   Particle_values
     * @endcode
     *
     * `DatatypeSize` can have one of two values:
     *  - 4, float;
     *  - 8, double.
     *
     * `Number_of_records_per_line` gives the data count for each line of
     * the `Particle_values`. For example :
     *  - 4, the particle values are `X Y Z Q`;
     *  - 8, the particle values are `X Y Z Q  P FX FY FZ`.
     *
     * @tparam FReal
     */
    template<class FReal>
    class FFmaGenericWriter
    {
      protected:
        std::fstream* m_file;   ///< the stream used to write the file.

        bool m_binaryFile;   ///<  if true the file is in binary mode.

        bool m_verbose;   ///<  verbose mode.

      public:
        /**
         * @brief Construct a new FFmaGenericWriter object
         *
         * This constructor opens a file to be written to.
         *
         * - The opening mode is guessed from the file extension : `.fma` will open
         * in ASCII mode, `.bfma` will open in binary mode.
         *
         * @param filename the name of the file to open.
         */
        FFmaGenericWriter(std::string const& filename, const bool verbose = true)
          : m_binaryFile(false)
          , m_verbose{verbose}
        {
            if(m_verbose)
            {
                std::cout << "FFmaGenericWriter filename " << filename << std::endl;
            }
            std::string ext(".bfma");
            // open particle file
            if(filename.find(".bfma") != std::string::npos)
            {
                m_binaryFile = true;
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::out | std::ios::binary);
            }
            else if(filename.find(".fma") != std::string::npos)
            {
                this->m_file = new std::fstream(filename.c_str(), std::ifstream::out);
                this->m_file->precision(std::numeric_limits<FReal>::digits10);
            }
            else
            {
                std::cout << "filename " << filename << " find fma " << filename.find(".fma") << " "
                          << std::string::npos << std::endl;
                std::cout << "Output file not allowed only .fma or .bfma extensions" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            // test if open
            if(!this->m_file->is_open())
            {
                std::cerr << "File " << filename << " not opened! " << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if(m_verbose)
            {
                std::cout << "FFmaGenericWriter file " << filename << " opened" << std::endl;
            }
        }

        void close()
        {
            m_file->close();
            delete m_file;
            m_file = nullptr;
        }

        /**
         * @brief Destroy the FFmaGenericWriter object.
         *
         * Default destructor, closes the file.
         */
        virtual ~FFmaGenericWriter()
        {
            if(m_file != nullptr)
            {
                m_file->close();
            }
        }

        /**
         * @brief To know if file is open and ready to read.
         *
         * @return true
         * @return false
         */
        inline auto isOpen() const -> bool { return this->m_file->is_open() && !this->m_file->eof(); }

        /**
         * @brief To know if opened file is in binary mode.
         *
         * @return true
         * @return false
         */
        inline auto isBinary() const -> bool { return this->m_binaryFile; }

        /**
         * @brief Writes the header of FMA file.
         *
         * Should be used if we write the particles with writeArrayOfReal method
         *
         * @tparam PointType  The type of a point
         * @tparam IntType1   The type of a nbParticles
         * @tparam IntType1  The type of a theother integer variables
         * @param centerOfBox      The center of the Box (FPoint<FReal, dimension> class)
         * @param boxWidth         The width of the box
         * @param nbParticles      Number of particles in the box (or to save)
         * @param dataType         Size of the data type of the values in particle
         * @param nbDataPerRecord  Number of record/value per particle
         * @param dimension.       The dimension of the problem 
         * @param nb_input_values. The number of input values in a particle
         */
        template<class PointType, typename IntType1, typename IntType2>
        auto writeHeader(const PointType& centerOfBox, const FReal& boxWidth, const IntType1& nbParticles,
                         const IntType2& dataType, const IntType2& nbDataPerRecord, const IntType2& dimension,
                         const IntType2& nb_input_values) -> void
        {
            std::array<unsigned int, 4> typeFReal = {
              static_cast<unsigned int>(dataType), static_cast<unsigned int>(nbDataPerRecord),
              static_cast<unsigned int>(dimension), static_cast<unsigned int>(nb_input_values)};

            FReal x = boxWidth * FReal(0.5);
            if(this->m_binaryFile)
            {
                this->writerBinaryHeader(&(centerOfBox[0]), x, nbParticles, typeFReal.data(), 4);
            }
            else
            {
                this->writerAscciHeader(centerOfBox, x, nbParticles, typeFReal.data(), 4);
            }
            if(m_verbose)
            {
                std::cout << "   Datatype ";
                for(int i = 0; i < 4; ++i)
                {
                    std::cout << " " << typeFReal[i];
                }
                std::cout << '\n';
                std::cout << "   nbParticles: " << nbParticles << std::endl
                          << "   Box width:   " << boxWidth << std::endl
                          << "   Center:      ";
                for(auto e: centerOfBox)
                {
                    std::cout << e << " ";
                }
                std::cout << std::endl << std::flush;
            }
        }
        template<typename IntType1, typename IntType2>
        auto writeHeader(const FReal* centerOfBox, const FReal& boxWidth, const IntType1& nbParticles,
                         const IntType2& dataType, const IntType2& nbDataPerRecord, const IntType2& dimension,
                         const IntType2& nb_input_values) -> void
        {
            std::array<unsigned int, 4> typeFReal = {
              static_cast<unsigned int>(dataType), static_cast<unsigned int>(nbDataPerRecord),
              static_cast<unsigned int>(dimension), static_cast<unsigned int>(nb_input_values)};

            FReal x = boxWidth * FReal(0.5);
            if(this->m_binaryFile)
            {
                this->writerBinaryHeader(centerOfBox, x, nbParticles, typeFReal.data(), 4);
            }
            else
            {
                this->writerAscciHeader(centerOfBox, x, nbParticles, typeFReal.data(), 4);
            }
            if(m_verbose)
            {
                std::cout << "   Datatype ";
                for(int i = 0; i < 4; ++i)
                {
                    std::cout << " " << typeFReal[i];
                }
                std::cout << '\n';
                std::cout << "   nbParticles: " << nbParticles << std::endl
                          << "   Box width:   " << boxWidth << std::endl
                          << "   Center:      ";
                for(int i = 0; i < typeFReal[2]; ++i)
                {
                    std::cout << centerOfBox[i] << " ";
                }
                std::cout << std::endl << std::flush;
            }
        }
        /**
         * @brief
         *
         * @tparam PointType
         * @param centerOfBox
         * @param boxWidth
         * @param nbParticles
         * @param dataType
         * @param nbDataPerRecord
         * @param dimension
         * @param nb_input_values
         */
        template<class PointType>
        auto writeHeaderOld(const PointType& centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                            const unsigned int dataType, const unsigned int nbDataPerRecord,
                            const unsigned int dimension, const unsigned int nb_input_values) -> void
        {
            std::array<unsigned int, 4> typeFReal = {dataType, nbDataPerRecord, dimension, nb_input_values};
            FReal x = boxWidth * FReal(0.5);
            if(this->m_binaryFile)
            {
                this->writerBinaryHeader(&centerOfBox[0], x, nbParticles, typeFReal.data(), 2);
            }
            else
            {
                this->writerAscciHeader(centerOfBox, x, nbParticles, typeFReal.data(), 2);
            }
        }

        /**
         *  @brief Write an array of data in a file Fill
         *
         * @param dataToWrite array of particles of type FReal
         * @param nbData number of data per particle
         * @param N number of particles
         *
         *   The size of the array is N*nbData
         *
         *   example
         * @code
         * FmaRParticle * const particles = new FmaRParticle[nbParticles];
         * memset(particles, 0, sizeof(FmaRParticle) * nbParticles) ;
         * ...
         * FFmaGenericWriter<FReal> writer(filenameOut) ;
         * Fwriter.writeHeader(Centre,BoxWith, nbParticles,*particles) ;
         * Fwriter.writeArrayOfReal(particles, nbParticles);
         * @endcode
         */
        template<typename int1Type, typename int2Type>
        auto writeArrayOfReal(const FReal* dataToWrite, const int1Type nbData, const int2Type N) -> void
        {
            if(m_binaryFile)
            {
                m_file->write((const char*)(dataToWrite), N * nbData * sizeof(FReal));
            }
            else
            {
                this->m_file->precision(std::numeric_limits<FReal>::digits10);

                std::size_t k = 0;
                for(std::size_t i = 0; i < N; ++i)
                {
                    // std::cout << "i "<< i << "  ";
                    for(int jj = 0; jj < nbData; ++jj, ++k)
                    {
                        (*this->m_file) << dataToWrite[k] << "    ";
                        // std::cout      << dataToWrite[k]<< "  ";
                    }
                    (*this->m_file) << std::endl;
                    // std::cout <<std::endl;
                }
                // std::cout << "END"<<std::endl;
            }
        }

        /**
         * @brief  Write all particles (position, input values, output values)
         *  inside the octree in fma format into a file.
         *
         *   example
         * @code
         *  group_tree_type tree(TreeHeight, SubTreeHeight, BoxWidth, CenterOfBox);
         * ...
         * FFmaGenericWriter<FReal> writer(filenameOut) ;
         * Fwriter.writeDataFromOctree(&tree, nbParticles);
         * @endcode
         *
         * @tparam TreeType
         * @param[in] tree Octree that contains the particles in the leaves
         * @param[in] number_particles number of particles
         */
        template<class TreeType>
        auto writeDataFromTree(const TreeType& tree, const std::size_t& number_particles) -> void
        {
            constexpr std::size_t dimension = TreeType::leaf_type::dimension;
            constexpr std::size_t nb_input_elements = TreeType::leaf_type::particle_type::inputs_size;
            constexpr std::size_t nb_output_elements = TreeType::leaf_type::particle_type::outputs_size;
            constexpr std::size_t nb_elt_per_par = dimension + nb_input_elements + nb_output_elements;
            if(m_verbose)
            {
                std::cout << "Dimension: " << dimension << std::endl;
                std::cout << "Number of input values: " << nb_input_elements << std::endl;
                std::cout << "Number of output values: " << nb_output_elements << std::endl;
                std::cout << "nb_elt_per_par " << nb_elt_per_par << std::endl;
            }
            //
            using value_type = typename TreeType::leaf_type::value_type;
            using particles_t = std::array<value_type, nb_elt_per_par>;
            std::vector<particles_t> particles(number_particles);
            //
            int pos = 0;
            scalfmm::component::for_each_mine_leaf(tree.begin_mine_leaves(), tree.end_mine_leaves(),
                                                   [&pos, &particles](auto& leaf)
                                                   {
                                                       for(auto const& it_p: leaf)
                                                       {
                                                           auto& particles_elem = particles[pos++];
                                                           const auto& p =
                                                             typename TreeType::leaf_type::particle_type(it_p);
                                                           //
                                                           int i = 0;
                                                           const auto points = p.position();
                                                           for(int k = 0; k < dimension; ++k, ++i)
                                                           {
                                                               particles_elem[i] = points[k];
                                                           }
                                                           // get inputs
                                                           for(int k = 0; k < nb_input_elements; ++k, ++i)
                                                           {
                                                               particles_elem[i] = p.inputs(k);
                                                           }
                                                           // get outputs
                                                           for(int k = 0; k < nb_output_elements; ++k, ++i)
                                                           {
                                                               particles_elem[i] = p.outputs(k);
                                                           }
                                                       }
                                                   });
            //
            // write the particles
            const auto& centre = tree.box_center();
            this->writeHeader(centre, tree.box_width(), number_particles, static_cast<std::size_t>(sizeof(value_type)),
                              nb_elt_per_par, centre.dimension, nb_input_elements);
            this->writeArrayOfReal(particles.data()->data(), nb_elt_per_par, number_particles);
        }

        /**
         * @brief
         *
         *  writeDataFrom write all data from the container in fma format
         *
         *  How to get automatically double from container
         *
         * @tparam ContainerType
         * @tparam PointType
         * @tparam ValueType
         * @param values the contnaire of particles
         * @param number_particles.  number of particles to write
         * @param center
         * @param box_width
         */
        template<class ContainerType, class PointType, typename ValueType>
        auto writeDataFrom(ContainerType& values, const int& number_particles, const PointType& center,
                           const ValueType box_width) -> void
        {
            // get the number of elements per particles in the container build with tuples.
            using particle_type = typename ContainerType::value_type;
            constexpr std::size_t nb_elt_per_par = meta::tuple_size_v<particle_type>;
            static constexpr std::size_t dimension = particle_type::dimension_size;

            // Not good output_values are put in input_values
            constexpr std::size_t nb_input_per_par = particle_type::inputs_size;
            ///  @todo check for different input and output types (double versus complexe)
            using data_type = typename particle_type::outputs_value_type;
            //
            using particles_t = std::array<data_type, nb_elt_per_par>;
            std::vector<particles_t> particles(number_particles);

#pragma omp parallel for shared(particles)
            for(auto it_p = std::begin(values); it_p < std::end(values); ++it_p)
            {
                particles_t particles_elem{};
                int pos = std::distance(std::begin(values), it_p);
                // fill the curent particles inputs outputs in the vector

                scalfmm::meta::for_each(particles_elem, *it_p, [](const auto& tuple_elm) { return tuple_elm; });

                particles[pos] = particles_elem;
            };
            //  write the particles
            // Here we need to separate input from output variables - no tools yet
            this->writeHeader(center, box_width, number_particles, static_cast<std::size_t>(sizeof(data_type)),
                              nb_elt_per_par, dimension, nb_input_per_par);
            this->writeArrayOfReal(particles.data()->data(), nb_elt_per_par, number_particles);
        }
        /**
         * @brief
         *
         *  writeDataFrom write all data from the std::vector of particles in fma format
         *
         *  How to get automatically double from container
         *
         * @tparam ContainerType
         * @tparam PointType
         * @tparam ValueType
         * @param values the vector of particles
         * @param center the center of the box
         * @param box_width th width of the box
         */
        template<class ParticleType, class PointType, typename ValueType>
        auto writeDataFrom(std::vector<ParticleType>& values, const PointType& center, const ValueType box_width)
          -> void
        {
            // get the number of elements per particles in the container build with tuples.
            using particle_type = ParticleType;
            static constexpr std::size_t dimension = particle_type::dimension_size;
            static constexpr std::size_t outputs_size = particle_type::outputs_size;
            static constexpr std::size_t inputs_size = particle_type::inputs_size;
            constexpr std::size_t nb_elt_per_par = scalfmm::meta ::particle_size_without_variables<particle_type>::size;
            ///  @todo check for different input and output types (double versus complexe)
            using data_type = typename particle_type::outputs_value_type;
            using particles_t = std::array<data_type, nb_elt_per_par>;
            // Not good output_values are put in input_values
            constexpr std::size_t nb_input_per_par = particle_type::inputs_size;

            auto number_particles = values.size();
            // if constexpr(nb_elt_per_par < scalfmm::meta ::tuple_size_v<particle_type>)
            if constexpr(sizeof(particle_type) > nb_elt_per_par * sizeof(ValueType))
            //
            {
                // std::cout << " I have variables\n";
                std::vector<particles_t> particles(number_particles);
#pragma omp parallel for shared(particles, values)
                for(std::size_t i = 0; i < values.size(); ++i)
                {
                    auto ptr_data = &(values[i].position()[0]);
                    std::copy(ptr_data, ptr_data + nb_elt_per_par, particles[i].data());
                }
                //  write the particles
                // Here we need to separate input from output variables - no tools yet
                this->writeHeader(center, box_width, number_particles, static_cast<std::size_t>(sizeof(data_type)),
                                  nb_elt_per_par, dimension, nb_input_per_par);
                this->writeArrayOfReal(particles.data()->data(), nb_elt_per_par, number_particles);
            }
            else
            {
                // std::cout << nb_elt_per_par << " . " << scalfmm::meta ::tuple_size_v<particle_type> << std::endl;
                this->writeHeader(center, box_width, number_particles, static_cast<std::size_t>(sizeof(data_type)),
                                  nb_elt_per_par, dimension, nb_input_per_par);
                // std::cout << " " << &(values[0].position()[0]) << "  " << values.data() << std::endl;
                auto ptr_data = &(values[0].position()[0]);
                // int k{0};
                // for(int i = 0; i < 3; ++i)
                // {
                //     std::cout << i << " part " << values[i] << " array ";
                //     for(int j = 0; j < nb_elt_per_par; ++j, ++k)
                //     {
                //         std::cout << ptr_data[k] << " ";
                //     }
                //     std::cout << std::endl;
                // }
                this->writeArrayOfReal(ptr_data, nb_elt_per_par, number_particles);
            }
        }

      protected:
        /**
         * @brief Write the header of the file for fma format
         *
         * @tparam PointType  The type of the point 
         * @param centerOfBox The centre of the box
         * @param boxWidth the size of the Box
         * @param nbParticles Number of particles to write
         * @param typeFReal array of size 4 containing {dataType, nbDataPerRecord, dimension, nb_input_values};
         * @param nbVal Number of values for one particles (dimension + nb_input_values + nb_output_values)
         */
        template<class PointType>
        auto writerAscciHeader(const PointType& centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                               const unsigned int* typeFReal, const unsigned int nbVal) -> void
        {
            this->m_file->precision(std::numeric_limits<FReal>::digits10);
            // Line 1
            (*this->m_file) << typeFReal[0];
            for(unsigned int i = 1; i < nbVal; ++i)
            {
                (*this->m_file) << "  " << typeFReal[i];
            }
            (*this->m_file) << '\n';
            // Line 2
            (*this->m_file) << nbParticles << "   " << boxWidth;
            for(std::size_t i = 0; i < centerOfBox.size(); ++i)
            {
                (*this->m_file) << "  " << centerOfBox[i];
                ;
            }

            (*this->m_file) << '\n';
        }
        /**
         * @brief Write the header of the file for fma format
         *
         * @param centerOfBox The centre of the box
         * @param boxWidth the size of the Box
         * @param nbParticles Number of particles to write
         * @param typeFReal array of size 4 containing {dataType, nbDataPerRecord, dimension, nb_input_values};
         * @param nbVal Number of values for one particles (dimension + nb_input_values + nb_output_values)
         */
        auto writerAscciHeader(const FReal* centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                               const unsigned int* typeFReal, const unsigned int nbVal) -> void
        {
            this->m_file->precision(std::numeric_limits<FReal>::digits10);
            // Line 1
            (*this->m_file) << typeFReal[0];
            for(unsigned int i = 1; i < nbVal; ++i)
            {
                (*this->m_file) << "  " << typeFReal[i];
            }
            (*this->m_file) << '\n';
            // Line 2
            (*this->m_file) << nbParticles << "   " << boxWidth;
            for(std::size_t i = 0; i < typeFReal[2]; ++i)
            {
                (*this->m_file) << "  " << centerOfBox[i];
                ;
            }

            (*this->m_file) << '\n';
        }
        /**
         * @brief Write the header of the file for bfma format
         *
         * @tparam PointType  The type of a point 
         * @param centerOfBox the centre of the box (point<Freal, typeFReal[2]>)
         * @param boxWidth the size of the Box
         * @param nbParticles Number of particles to write
         * @param typeFReal array of size 4 containing {dataType, nbDataPerRecord, dimension, nb_input_values};
         * @param nbVal Number of values for one particles (dimension + nb_input_values + nb_output_values)
         */
        template<class PointType>
        auto writerBinaryHeader(PointType const& centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                                const unsigned int* typeFReal, const unsigned int nbVal) -> void
        {
            m_file->seekg(std::ios::beg);
            m_file->write((const char*)typeFReal, nbVal * sizeof(unsigned int));
            if(typeFReal[0] != sizeof(FReal))
            {
                std::cout << "writerBinaryHeader: Size of elements in part file " << typeFReal[0]
                          << " is different from size of FReal " << sizeof(FReal) << std::endl;
                std::exit(EXIT_FAILURE);
            }
            else
            {
                m_file->write((const char*)&(nbParticles), sizeof(std::size_t));
                // std::cout << "nbParticles "<< nbParticles<<std::endl;
                m_file->write((const char*)&(boxWidth), sizeof(boxWidth));
                if(nbVal == 2)   // old Format
                {
                    m_file->write((const char*)(&(centerOfBox[0])), sizeof(FReal) * 3);
                }
                else
                {
                    m_file->write((const char*)(&(centerOfBox[0])), sizeof(FReal) * typeFReal[2]);
                }
            }
        }
        /**
         * @brief Write the header of the file for bfma format
         *
         * @param centerOfBox A pointer on the position of the centre of the box (size typeFReal[2])
         * @param boxWidth the size of the Box
         * @param nbParticles Number of particles to write
         * @param typeFReal array of size 4 containing {dataType, nbDataPerRecord, dimension, nb_input_values};
         * @param nbVal Number of values for one particles (dimension + nb_input_values + nb_output_values)
         */
        auto writerBinaryHeader(const FReal* centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                                const unsigned int* typeFReal, const unsigned int nbVal) -> void
        {
            m_file->seekg(std::ios::beg);
            m_file->write((const char*)typeFReal, nbVal * sizeof(unsigned int));
            if(typeFReal[0] != sizeof(FReal))
            {
                std::cout << "writerBinaryHeader: Size of elements in part file " << typeFReal[0]
                          << " is different from size of FReal " << sizeof(FReal) << std::endl;
                std::exit(EXIT_FAILURE);
            }
            else
            {
                m_file->write((const char*)&(nbParticles), sizeof(std::size_t));
                m_file->write((const char*)&(boxWidth), sizeof(boxWidth));
                if(nbVal == 2)   // old Format
                {
                    m_file->write((const char*)(centerOfBox), sizeof(FReal) * 3);
                }
                else
                {
                    m_file->write((const char*)(centerOfBox), sizeof(FReal) * typeFReal[2]);
                }
            }
        }
    };
}   // namespace scalfmm::io
#endif   // SCALFMM_TOOLS_FMA_LOADER_HPP
