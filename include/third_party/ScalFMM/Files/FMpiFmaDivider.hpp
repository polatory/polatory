#ifndef _FMPIFMADIVIDER_HPP_
#define _FMPIFMADIVIDER_HPP_

#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <stdexcept>
#include "../Files/FFmaGenericLoader.hpp"



#include "../Containers/FCoordinateComputer.hpp"
#include "../Utils/FPoint.hpp"

#include "../Utils/FLog.hpp"

#include "../Utils/FRepeatAction.hpp"

/** \brief Loads an FMA particle file and splits it for MPI processing.
 *
 * This is meant to be used to split an FMA particle file and save each part in
 * a different file for MPI processes to pick up later. The aim is to distribute
 * the particles in a way that will balance the computing load of all processes
 * during the FMM.
 *
 * The tree is not actually built. The user chooses the tree-level at which he
 * wants to split the file. The generated files can then be loaded with an
 * FFmaGenericLoader.
 *
 * Each process posseses the fraction of the tree that goes from its root to the
 * particles that it loaded.
 *
 * **2D example for a division level of 2, a tree of height 4 and 3 processes**
 *
 * The space is divided according to the division level.
 *
 *          2D space                  level 1                  level 2
 *      |* * * * *   * *|   ->   |* * * *|*   * *|   ->   |* *|* *|*  |* *|
 *
 * Then the divisions are associated with processors according to particle count
 * and written to the file system.
 *
 *      |* *|* *|*  |* *|   ->   |* *|* *|*   * *|
 *        0   1   2   2            0   1     2
 *
 * The files are subsequently loaded into a tree.
 *
 *                Full tree                                        Proc 0 tree
 *                                                                       _
 *                  *                                                  *  |
 *               /     \                                            /     | nodes duplicated
 *            *           *                                      *        | between processes
 *      ____/___\_______/___\____ <-- division level       ____/__       _|
 *      |  *  |  *  |  *     *  |                     =>   |  *  |        |
 *      | / \ | / \ | /     / \ | <-- sub-trees            | / \ |        | nodes exclusive to proc 0
 *      |*   *|*   *|*     *   *| <-- particles            |*   *|       _|
 *         0     1        2                                   0
 *
 * \tparam FReal The floating point representation (usually double).
 * 
 */
template <typename FReal>
class FMpiFmaDivider {
public:
    /// Policy for entites that are dispatched.
    /**  - PARTICLES to dispatch by counting particles.
     *   - BOXES to dispatch by counting Morton index boxes. */
    enum DispatchPolicy {PARTICLES = 0, BOXES = 1};

private:

    /** \brief Particle structure.
     *
     * This structure is used to store particles read from an input file. It is
     * designed to ease sorting according to the Morton index of the tree cells
     * without building it.
     *
     * \note All particles are stored in the same box space which is divided in
     * 8^_divisionLevel cells.
     *
     * \todo Merge with / replace by FmaRWParticle <8,8> ?
     */
    class FmaParticle {
    public:
        /// Center of the particles' box
        static FPoint<FReal> _boxCenter;
        /// Width of the particles' box
        static FReal _boxWidth;
        /// Division level of the particles' box
        static int   _divisionLevel;
        
    private:
        /// Position and physical values of the particle
        /** Stored in the following order. Q is the physical value of the
         *  particle, P is its potential, FX, FY and FZ are the components of the
         *  forces applying to it.
         *
         *     x y z Q [P FX FY FZ]
         */
        FReal data[8];

        /// Piece of data count for this particle (4 <= dataCount <= 8)
        unsigned int dataCount;

        /// Enumeration to avoid calling the delegated constructor with overloading.
        enum do_not_overload {};
        /// Delegated constructor that automatically fills the data array.
        /** See
         *  - [How can I prevent a variadic constructor from being prefered to the copy constructor ?]
         *    (http://stackoverflow.com/q/13937873)
         *  - [Can one overload a constructor with a private constructor with the same parameters ?]
         *    (http://stackoverflow.com/q/14679299)
         */
        template<typename...Args>
        FmaParticle(do_not_overload, Args&&... args) :
            data{args...},
            dataCount{sizeof...(Args)}
            {}

    public:
        /// Contructor
        FmaParticle(FPoint<FReal>& pos, FReal& val) :
            FmaParticle(do_not_overload(), pos.getX(), pos.getY(), pos.getZ(), val)
            { }
        
        /// Default constructor
        FmaParticle() = default;
        /// Default copy constructor
        FmaParticle(const FmaParticle& other) = default;
        /// Default move constructor
        FmaParticle(FmaParticle&& other) = default;


        /// Get containing cell Morton index.
        /** The index is computed from #position relative to #_divisionLevel,
         * #_boxWidth and #_boxCenter.
         * \return The containing cell Morton index
         */
        MortonIndex getMortonIndex() const {
            return FCoordinateComputer::
                GetCoordinateFromPosition(_boxCenter, _boxWidth,
                                          _divisionLevel, getPosition()).
                getMortonIndex();
            
        }

        /// Particle's position accessor.
        FPoint<FReal> getPosition() const {
            return FPoint<FReal>(data[0], data[1], data[2]);
        }

        /// Particle's physical value accessor.
        FReal getPhysicalValue() const {
            return data[3];
        }

        /// Particle's forces accessor.
        /**\return
         * An FReal array if #dataCount >= 6; nullptr otherwise.
         * The returned array is not garanteed to be 3 elements long. Use
         * getDataCount() - 5 to compute the length.
         */
        const FReal* getForces() const {
            return dataCount >= 6 ? data+5 : nullptr;
        }

        /// Particle's potential accessor.
        /**\returns
         * The potential if #dataCount >= 5; 0 otherwise.
         */
        FReal getPotential() const {
            return dataCount >= 5 ? data[4] : 0;
        }

        /// Particle's raw #data accessor.
        FReal* getPtrFirstData() {
            return data;
        }

        /// Particle's #data length.
        unsigned int getDataCount() {
            return dataCount;
        }

        /// Particle's #data length.
        unsigned int getWriteDataNumber() {
            return getDataCount();
        }

        /// Equality operator
        bool operator==(const FmaParticle& other) const {
            return getPosition() == other.getPosition() &&
                getPhysicalValue() == other.getPhysicalValue();
        }

        /// Strictly less than operator
        bool operator<(const FmaParticle& other) const {
            return getMortonIndex() < other.getMortonIndex();
        }

        /// Output stream operator
        friend std::ostream& operator<<(std::ostream& os, const FmaParticle& p) {
            for( int i = 0; i < p.dataCount; i++) {
                os << p.data[i] << " ";
            }
            return os;
        }
    };

    /// The input file name
    std::string _filename;

    /// The output files basename
    std::string _outputBasename;

    /// The output files extension
    std::string _outputExt;

    /// The number of parts to split the file into
    int _splitCount; 

    /// The level of division of space
    int _divisionLevel;

    /// A set to sort the particles using their Morton index
    std::multiset<FmaParticle> _particles;

    /// Particle count as returned by reading the file header with the loader
    FSize _particleCount {0};

    /// Data type as returned by reading the file header with the loader
    unsigned int _dataType {sizeof(FReal)};

    /// Record count per line as returned by reading the file header with the loader
    unsigned int _nbRecordsPerLine {4};

    /// What to count to choose file content
    DispatchPolicy _dispatchPolicy = PARTICLES;

public:

    /** \brief Constructor
     *
     * Builds the particle divider and opens the loader. The output files
     * basename and extension are extracted from the input file name.
     *
     * \param filename[in] The particle input file.
     * \param splitcount[in] The number of parts to split the input file into.
     * \param divisionLevel[in] The level in the to-be tree at which to do the division.
     */
    FMpiFmaDivider(const std::string filename,
                   int splitCount,
                   int divisionLevel,
                   DispatchPolicy dispatchPolicy = PARTICLES) :
        FMpiFmaDivider(filename, 
                       filename.substr(0,filename.find_last_of(".")),
                       filename.substr(filename.find_last_of(".")),
                       splitCount,
                       divisionLevel,
                       dispatchPolicy ) {
        }

    /** \brief Constructor
     *
     * Builds the particle divider and opens the loader.
     *
     * \param filename[in] The particle input file.
     * \param outputBasename[in] The basename for the output files.
     * \param splitcount[in] The number of parts to split the input file into.
     * \param divisionLevel[in] The level in the to-be tree at which to do the division.
     */
    FMpiFmaDivider(const std::string filename,
                   const std::string basename,
                   const std::string ext,
                   int splitCount,
                   int divisionLevel,
                   DispatchPolicy dispatchPolicy) :
        _filename(filename),
        _outputBasename(basename),
        _outputExt(ext),
        _splitCount(splitCount),
        _divisionLevel(divisionLevel),
        _dispatchPolicy(dispatchPolicy) {

        readFile();
        writeFiles();

    }


    /// Deleted copy constructor because of loader
    FMpiFmaDivider(const FMpiFmaDivider&) = delete;
    /// Deleted move constructor because of loader
    FMpiFmaDivider(const FMpiFmaDivider&&) = delete;


protected:
    /// Reads the particles from the loader and stores them sorted by Morton index.
    void readFile() {

        // The particle loader
        FFmaGenericLoader<FReal> loader(_filename);
        _particleCount = loader.getNumberOfParticles();
        _dataType = loader.getDataType();
        _nbRecordsPerLine = loader.getNbRecordPerline();

        // Set particles' space to enable comparisons.
        FmaParticle::_boxCenter = loader.getCenterOfBox();
        FmaParticle::_boxWidth = loader.getBoxWidth();
        FmaParticle::_divisionLevel = _divisionLevel;            


        FReal valueBuffer;
        FPoint<FReal> positionBuffer;
        
        int idxPart = 0;

        ////////////////////////
        FLOG( 
            FRepeatAction progress(
                [&]() -> bool {
                    std::cerr
                        << "     \rReading & sorting particles: "
                        << idxPart * 100 / _particleCount << "%" ;
                    return true; }, 200 ); 
            );
        ////////////////////////

        for ( idxPart = 0 ; idxPart < _particleCount; idxPart++) {
            loader.fillParticle(&positionBuffer, &valueBuffer);
            _particles.emplace(positionBuffer, valueBuffer);
        }

        ////////////////////////
        FLOG({                
                progress.stop();
                std::cerr << std::endl;
                std::cout << "@@ read-insert-particle-time:"
                          << progress.cumulated()
                          << std::endl;
            });
        ////////////////////////
    }

    /// Writes the output file based on #_outputBasename.
    void writeFiles() {

        std::vector<std::vector<FmaParticle> > filesParticles(_splitCount);

        // Temporary vector to store all particles with same Morton index
        std::vector<FmaParticle> container;

        using size_type = typename std::vector<FmaParticle>::size_type;
        
        size_type totalCount = _particleCount;
        size_type currentCount = 0;
        MortonIndex currentMortonIdx = 0;
        long long int currentFileIdx = 0;

        if( BOXES == _dispatchPolicy ) {
            totalCount = 1;
            for ( auto p :  _particles ) {
                if( p.getMortonIndex() != currentMortonIdx) {
                    totalCount++;
                    currentMortonIdx = p.getMortonIndex();
                }
            }
            currentMortonIdx = 0;
        }

        // FIND OUT PARTICLES' FILE ////////////////////////////////////////////
        FLOG( // Choose file progress log
            FRepeatAction progress(
                [&]() -> bool {
                    std::cerr
                        << "     \rCreating particle-files association: "
                        << currentCount * 100 / totalCount
                        << "%" ;
                    return true; }, 200 ); );
        ////////////////////////

        while(_particles.begin() != _particles.end()) {
            /* Particles are sorted, we pull out all those with the same Morton
             * index and put them in the temporary container. */
            while( _particles.begin() != _particles.end()
                   && (*_particles.begin()).getMortonIndex() == currentMortonIdx) {
                container.push_back(*(_particles.begin()));
                _particles.erase(_particles.begin());
            }


            // File number
            currentFileIdx = currentCount * _splitCount
                / totalCount;

            // Set particles to file.
            for(FmaParticle p : container) {
                filesParticles[currentFileIdx].push_back(p);
            }

            if( BOXES == _dispatchPolicy ) {
                currentCount++;
            } else if ( PARTICLES == _dispatchPolicy ) {
                currentCount += container.size();
            }
            container.clear();
            currentMortonIdx = (*_particles.begin()).getMortonIndex();
        }

        ////////////////////////
        FLOG({
                progress.stop();
                std::cerr << std::endl;
                std::cout << "@@ distribute-particle-time:"
                          << progress.cumulated()
                          << std::endl;
            });
        ////////////////////////

        currentFileIdx = 0;
        currentCount = 0;



        // WRITE FILES /////////////////////////////////////////////////////////
        FLOG( /// Write files progress log
            std::cerr << "\rWriting particles to files... ";
            progress.setFunction(
                [&]() -> bool {
                    std::cerr
                        << "     \rWriting particles to file "
                        << currentFileIdx
                        << ": "
                        << currentCount * 100 / _particleCount
                        << "%" ;
                    return true; } );
            progress.start();
            );
        ////////////////////////

        // Write common file header
        {
            FFmaGenericWriter<FReal>
                commonWriter(_outputBasename + ".main" + _outputExt);
            commonWriter.writeHeader(FmaParticle::_boxCenter,
                                     FmaParticle::_boxWidth,
                                     _particleCount,
                                     _dataType,
                                     _nbRecordsPerLine);
        }

        // Write files
        for ( std::vector<FmaParticle>& filecontent : filesParticles ) {
            // Open the current file writer
            FFmaGenericWriter<FReal>
                writer(_outputBasename + "." + std::to_string(currentFileIdx)
                       + _outputExt);

            writer.writeHeader(FmaParticle::_boxCenter,
                               FmaParticle::_boxWidth,
                               filecontent.size(),
                               _dataType,
                               _nbRecordsPerLine);

            for( FmaParticle p : filecontent ) {
                writer.writeArrayOfReal(p.getPtrFirstData(),
                                        p.getWriteDataNumber(), 1);

                FLOG(currentCount++);
            }
            currentFileIdx++;
        }

        ////////////////////////
        FLOG({
                progress.stop();
                std::cerr << std::endl;
                std::cout << "@@ write-particle-time:"
                          << progress.cumulated()
                          << std::endl;
            });
        ////////////////////////

    }


};


template<typename FReal>
FPoint<FReal> FMpiFmaDivider<FReal>::FmaParticle::_boxCenter = FPoint<FReal>{0,0,0};
template<typename FReal>
FReal FMpiFmaDivider<FReal>::FmaParticle::_boxWidth = 1;
template<typename FReal>
int FMpiFmaDivider<FReal>::FmaParticle::_divisionLevel = 4;



#endif
