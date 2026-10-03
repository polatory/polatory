// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tools/fma_dist_loader.hpp
// --------------------------------
#ifndef SCALFMM_TOOLS_FMA_MPI_LOADER_HPP
#define SCALFMM_TOOLS_FMA_MPI_LOADER_HPP

#include "scalfmm/tools/fma_loader.hpp"

#include <cpp_tools/parallel_manager/parallel_manager.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace scalfmm::io
{
    /**
     * @brief
     *
     * @tparam FReal
     * @tparam Dimension
     */
    template<class FReal, int Dimension = 3>
    class DistFmaGenericLoader : public FFmaGenericLoader<FReal, Dimension>
    {
      protected:
        using FFmaGenericLoader<FReal, Dimension>::m_nbParticles;
        using FFmaGenericLoader<FReal, Dimension>::m_file;
        using FFmaGenericLoader<FReal, Dimension>::m_typeData;
        using FFmaGenericLoader<FReal, Dimension>::m_verbose;
        using MPI_Offset = std::size_t;

        std::size_t m_local_number_of_particles;   ///<  Number of particles that the calling process will manage.

        std::size_t m_start;   ///<

        size_t m_headerSize;   ///< size of the header in byte

        const cpp_tools::parallel_manager::parallel_manager* m_parallelManager;   ///< a pointer on the parallel manager
        // MPI_Offset m_idxParticles;                 ///<

      public:
        /**
         * @brief Construct a new Dist Fma Generic Loader object
         *
         * @param inFilename
         * @param para
         * @param verbose
         */
        DistFmaGenericLoader(const std::string inFilename, const cpp_tools::parallel_manager::parallel_manager& para,
                             const bool verbose = false)
          : FFmaGenericLoader<FReal, Dimension>(inFilename, verbose)
          , m_local_number_of_particles(0)
          //   , m_idxParticles(0)
          , m_headerSize(0)
          , m_parallelManager(&para)
        {
            // the header is already read by the constructor of FFmaGenericLoader
            std::size_t bloc = m_nbParticles / FReal(m_parallelManager->get_num_processes());
            if(verbose)
            {
                std::cout << "bloc: " << bloc << " part " << m_nbParticles << " nP  "
                          << FReal(m_parallelManager->get_num_processes()) << std::endl;
            }
            // Determine the number of particles to read
            auto rank = m_parallelManager->get_process_id();
            std::size_t startPart = bloc * rank;
            std::size_t endPart =
              (rank + 1 == m_parallelManager->get_num_processes()) ? m_nbParticles : startPart + bloc;

            this->m_start = startPart;
            this->m_local_number_of_particles = endPart - startPart;
            if(verbose)
            {
                std::cout << " startPart " << startPart << " endPart " << endPart << std::endl;
                std::cout << "Proc " << m_parallelManager->get_process_id() << " will hold "
                          << m_local_number_of_particles << std::endl;
            }
            // Skip the header
            if(this->m_binaryFile)
            {
                // This is header size in bytes
                //   MEANING :      sizeof(FReal)+nbAttr, nb of parts, boxWidth+boxCenter
                m_headerSize = FFmaGenericLoader<FReal, Dimension>::get_header_size();

                // To this header size, we had the parts that belongs to proc on my left
                m_file->seekg(m_headerSize + startPart * m_typeData[1] * sizeof(FReal));
            }
            else
            {
                // First finish to read the current line
                m_file->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                for(std::size_t i = 0; i < startPart; ++i)
                {
                    // First finish to read the current line
                    m_file->ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
            }
        }

        /**
         * @brief Destroy the Dist Fma Generic Loader object
         *
         */
        ~DistFmaGenericLoader() {}

        /**
         * @brief Get the My Number Of Particles object
         *
         * @return auto
         */
        inline auto getMyNumberOfParticles() const { return m_local_number_of_particles; }

        /**
         * @brief Get the Start object
         *
         * @return auto
         */
        inline auto getStart() const { return m_start; }

        /**
         * @brief Given an index, get the one particle from this index.
         *
         * @param data
         * @param indexInFile
         */
        inline auto fill1Particle(FReal* data, std::size_t indexInFile) -> void
        {
            m_file->seekg(m_headerSize +
                          (int(indexInFile) * FFmaGenericLoader<FReal>::getNbRecordPerline() * sizeof(FReal)));
            m_file->read((char*)data, FFmaGenericLoader<FReal>::getNbRecordPerline() * sizeof(FReal));
        }
    };

    /**
     * @brief Writes a set of distributed particles to an FMA formated file.
     *
     * The file may be in ASCII or binary mode. The example below shows how to use the class.
     *
     * @code
     * // Instanciate the writer with a binary fma file (extension .bfma).
     * @endcode
     * ----------------------------------------
     * FMA is a simple format to store particles in a file. It is organized as follow.
     *
     * file
     *
     * @tparam FReal
     */
    template<class FReal>
    class DistFmaGenericWriter : public FFmaGenericWriter<FReal>
    {
      protected:
        const cpp_tools::parallel_manager::parallel_manager* m_parallelManager;   ///<

        // bool _writeDone; ///<

        int m_headerSize;   ///< size of the header in byte

        int m_nbDataTowritePerRecord;   ///<   number of data to write for one particle.

        std::size_t m_numberOfParticles;   ///< number of particle (global) to write in the file.

        using FFmaGenericWriter<FReal>::m_file;
#ifdef SCALFMM_USE_MPI
        /**
         * @brief MPI pointer on data file (write mode).
         *
         */
        MPI_File _mpiFile;
#endif
      public:
        using base_type = FFmaGenericWriter<FReal>;
        using value_type = FReal;
        /**
         * @brief Construct a new Dist Fma Generic Writer object
         *
         * This constructor opens a file to be written to.
         *
         * - The opening mode is guessed from the file extension : `.fma` will open
         * in ASCII mode, `.bfma` will open in binary mode.
         *
         * @param inFilename the name of the file to open.
         * @param para
         */
        DistFmaGenericWriter(const std::string inFilename, cpp_tools::parallel_manager::parallel_manager const& para,
                             const bool verbose = false)
          : FFmaGenericWriter<FReal>(inFilename, verbose)
          , m_parallelManager(&para)
          //   , _writeDone(false)
          , m_headerSize(0)
          , m_nbDataTowritePerRecord(8)
          , m_numberOfParticles(0)
        {
#ifdef SCALFMM_USE_MPI
            if(!this->isBinary())
            {
                if(para.io_master())
                {
                    std::cout << "DistFmaGenericWriter only works with binary file (.bfma)." << std::endl;
                }
                std::exit(EXIT_FAILURE);
            }
            auto comm = m_parallelManager->get_communicator();

            int fileIsOpen = MPI_File_open(comm.get_comm(), const_cast<char*>(inFilename.c_str()),
                                           MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &_mpiFile);
            // Is it open?
            if(fileIsOpen != MPI_SUCCESS)
            {
                std::cerr << "Cannot create parallel file, DistFmaGenericWriter constructor abort." << std::endl;
                std::exit(EXIT_FAILURE);
                return;
            }
#endif
        }

        /**
         * @brief Writes the header of FMA file.
         *
         * Should be used if we write the particles with writeArrayOfReal method
         *
         * @tparam PointType
         * @param centerOfBox      The center of the Box (PointType class)
         * @param boxWidth         The width of the box
         * @param nbParticles      Number of particles in the box (or to save)
         * @param dataType         Size of the data type of the values in particle
         * @param nbDataPerRecord  Number of record/value per particle
         * @param dimension
         * @param nb_input_values
         */
        template<class PointType>
        auto writeHeader(const PointType& centerOfBox, const FReal& boxWidth, const std::size_t& nbParticles,
                         const unsigned int dataType, const unsigned int nbDataPerRecord, const unsigned int dimension,
                         const unsigned int nb_input_values) -> void
        {
            //      * \code
            //      *   DatatypeSize  Number_of_record_per_line
            //      *   NB_particles  half_Box_width  Center_X  Center_Y  Center_Z
            //      *   Particle_values
            //      * \endcode
            std::array<unsigned int, 4> typeFReal = {dataType, nbDataPerRecord, dimension, nb_input_values};
            FReal x = boxWidth * FReal(0.5);
            m_headerSize = 0;
            m_nbDataTowritePerRecord = nbDataPerRecord;
            m_numberOfParticles = nbParticles;
            if(m_parallelManager->master())
            {
                auto half_Box_width = boxWidth * value_type(0.5);
                base_type::writerBinaryHeader(centerOfBox, half_Box_width, nbParticles, typeFReal.data(), 4);
                // std::cout << "centerOfBox " << centerOfBox << " half_Box_width " << half_Box_width << " nbParticles "
                //           << nbParticles << " dataType " << dataType << " nbDataPerRecord " << nbDataPerRecord
                //           << " dimension " << dimension << " nb_input_values " << nb_input_values << std::endl;
#ifdef SCALFMM_USE_MPI
                // for(auto a: typeFReal)
                // {
                //     std::cout << "typeFReal " << a << std::endl;
                // }
                int sizeType = 0;
                int ierr = 0;
                auto mpiInt64 = cpp_tools::parallel_manager::mpi::get_datatype<std::size_t>();
                auto mpiUInt = cpp_tools::parallel_manager::mpi::get_datatype<unsigned int>();
                auto mpiReal = cpp_tools::parallel_manager::mpi::get_datatype<FReal>();
                //
                if(typeFReal[0] != sizeof(FReal))
                {
                    std::cout << " ssss Size of elements in part file " << typeFReal[0]
                              << " is different from size of FReal " << sizeof(FReal) << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                //  ierr = MPI_File_write_at(_mpiFile, 0, typeFReal.data(), typeFReal.size(), mpiUInt,
                //  MPI_STATUS_IGNORE);
                MPI_Type_size(mpiUInt, &sizeType);
                m_headerSize += sizeType * typeFReal.size();
                //     ierr = MPI_File_write_at(_mpiFile, m_headerSize, &nbParticles, 1, mpiInt64, MPI_STATUS_IGNORE);
                MPI_Type_size(mpiInt64, &sizeType);
                m_headerSize += sizeType * 1;

                //                std::array<FReal, 1 + PointType::dimension> boxSim{};
                //                boxSim[0] = boxWidth;
                //                for(int d = 0; d < PointType::dimension; ++d)
                //                {
                //                    boxSim[d + 1] = centerOfBox[d];
                //                }
                //     ierr = MPI_File_write_at(_mpiFile, m_headerSize, &boxSim[0], 4, mpiReal, MPI_STATUS_IGNORE);
                MPI_Type_size(mpiReal, &sizeType);
                m_headerSize += sizeType * (1 + PointType::dimension);
                // Build the header offset
                // std::cout << " headerSize " << m_headerSize << std::endl;
                FFmaGenericWriter<FReal>::close();   // Why ?????
#endif
            }
#ifdef SCALFMM_USE_MPI
            auto comm = m_parallelManager->get_communicator();

            comm.bcast(&m_headerSize, 1, MPI_INT, 0);
            //  MPI_Bcast(&_headerSize, 1, MPI_INT, 0, m_parallelManager->global().getComm());
            // std::cout << "  _headerSize  " << m_headerSize << std::endl;
#endif

            //  MPI_File_close(&_mpiFile);
        }

        /**
         * @brief Destroy the Dist Fma Generic Writer object
         *
         */
        ~DistFmaGenericWriter() { /* MPI_File_close(&_mpiFile);*/ }

        /**
         * @brief Write all for all particles the position, physical values, potential and forces.
         *
         * @tparam TreeType
         * @param tree the octree
         * @param nbParticles number of particles.
         */
        template<typename TreeType>
        auto writeFromTree(const TreeType& tree, std::size_t const& nbParticles) -> void
        {
            // The header is already written
            static constexpr int dimension = TreeType::dimension;
            using int64 = int;

            int maxPartLeaf{0}, nbLocalParticles{0};
            scalfmm::component::for_each_mine_leaf(tree.begin_mine_leaves(), tree.end_mine_leaves(),
                                                   [&nbLocalParticles, &maxPartLeaf](auto& leaf)
                                                   {
                                                       auto n = static_cast<int>(leaf.size());
                                                       nbLocalParticles += n;
                                                       maxPartLeaf = std::max(maxPartLeaf, n);
                                                       ;
                                                   });
            // Build the offset for each processes
            int64 before{0};   // Number of particles before me (rank < myrank)
            int sizeRealType{0}, ierr{0};

            auto mpiReal = cpp_tools::parallel_manager::mpi::get_datatype<FReal>();
            MPI_Type_size(mpiReal, &sizeRealType);
            MPI_Datatype mpiInt64 = cpp_tools::parallel_manager::mpi::get_datatype<int64>();

            // #ifdef SCALFMM_USE_MPI
            auto comm = m_parallelManager->get_communicator();
            MPI_Scan(&nbLocalParticles, &before, 1, mpiInt64, MPI_SUM, comm);
            before -= nbLocalParticles;
            // std::cout << " nbLocalParticles  " << nbLocalParticles << " maxPartLeaf " << maxPartLeaf << " before "
            //           << before << std::endl
            //           << std::flush;
            //
            MPI_Offset offset = m_headerSize + sizeRealType * m_nbDataTowritePerRecord * before;
            // std::cout << " offset to write part  " << offset << std::endl;
            //
            // Write particles in file

            using value_type = typename TreeType::leaf_type::value_type;
            static constexpr int nb_elt_per_par =
              dimension + TreeType::particle_type::inputs_size + TreeType::particle_type::outputs_size;
            // std::cout << "nb_elt_per_par " << nb_elt_per_par << std::endl;
            using particles_t = std::array<value_type, nb_elt_per_par>;
            std::vector<particles_t> particles(nbLocalParticles);
            // std::vector<FReal> particles(maxPartLeaf * m_nbDataTowritePerRecord);

            //
            scalfmm::component::for_each_mine_leaf(
              tree.begin_mine_leaves(), tree.end_mine_leaves(),
              [this, &offset, &particles, &sizeRealType, &mpiReal](auto& leaf)
              {
                  int pos = 0;
                  auto nbPartsInLeaf = leaf.size();
                  //   std::cout << " leaf index " << leaf.index() << " nbpart " << leaf.size() << std::endl;
                  for(auto const& it_p: leaf)
                  {
                      auto& particles_elem = particles[pos++];
                      const auto& p = typename TreeType::leaf_type::particle_type(it_p);
                      //
                      int i = 0;
                      const auto points = p.position();
                      for(int k = 0; k < dimension; ++k, ++i)
                      {
                          particles_elem[i] = points[k];
                      }
                      // get inputs
                      for(int k = 0; k < TreeType::particle_type::inputs_size; ++k, ++i)
                      {
                          particles_elem[i] = p.inputs(k);
                      }
                      // get outputs
                      for(int k = 0; k < TreeType::particle_type::outputs_size; ++k, ++i)
                      {
                          particles_elem[i] = p.outputs(k);
                      }
                      //   std::cout << "      " << pos << " part  " << particles_elem << std::endl;
                  }
                  //   std::cout << " write to  ptr_data " << particles.data() << " size "
                  //             << static_cast<int>(m_nbDataTowritePerRecord * nbPartsInLeaf) << std::endl;
                  MPI_File_write_at(_mpiFile, offset, particles.data(),
                                    static_cast<int>(m_nbDataTowritePerRecord * nbPartsInLeaf), mpiReal,
                                    MPI_STATUS_IGNORE);
                  offset += sizeRealType * m_nbDataTowritePerRecord * nbPartsInLeaf;
                  //   std::cout << " next offset to write part  " << offset << std::endl;
                  //   std::cout << std::endl << std::flush;
              });

#ifdef SCALFMM_USE_MPI
            MPI_File_close(&_mpiFile);
#endif
        }

        //        /**
        //         *  Write all for all particles the position, physical values, potential and forces
        //         *
        //         * @param myOctree the octree
        //         * @param nbParticlesnumber of particles
        //         * @param nbLocalParticles number of local particles (on the MPI processus
        //         * @param mortonLeafDistribution the morton distribution of the leaves (this is a vector of size 2*
        //         the number
        //         * of MPI processes
        //         *
        //         */
        //        template<class TreeType>
        //        void writeDistributionOfParticlesFromOctree(TreeType& myOctree, const std::size_t& nbParticles,
        //                                                    const std::size_t& nbLocalParticles,
        //                                                    const std::vector<MortonIndex>& mortonLeafDistribution)
        //        {
        //            //
        //            // Write the header
        //            int sizeType = 0, ierr = 0;
        //            FReal tt{};

        //            MPI_Datatype mpistd::size_t_t = m_parallelManager->GetType(nbParticles);
        //            MPI_Datatype mpiFReal_t = m_parallelManager->GetType(tt);
        //            MPI_Type_size(mpiFReal_t, &sizeType);
        //            int myRank = m_parallelManager->global().processId();
        //            _headerSize = 0;
        //            //
        //            unsigned int typeFReal[2] = {sizeof(FReal), static_cast<unsigned int>(_nbDataTowritePerRecord)};
        //            if(myRank == 0)
        //            {
        //                ierr = MPI_File_write_at(_mpiFile, 0, &typeFReal, 2, MPI_INT, MPI_STATUS_IGNORE);
        //            }
        //            MPI_Type_size(MPI_INT, &sizeType);
        //            _headerSize += sizeType * 2;
        //            if(myRank == 0)
        //            {
        //                ierr = MPI_File_write_at(_mpiFile, _headerSize, const_cast<std::size_t*>(&nbParticles), 1,
        //                mpistd::size_t_t,
        //                                         MPI_STATUS_IGNORE);
        //            }
        //            MPI_Type_size(mpistd::size_t_t, &sizeType);
        //            _headerSize += sizeType * 1;
        //            auto centerOfBox = myOctree.getBoxCenter();
        //            FReal boxSim[4] = {myOctree.getBoxWidth() * 0.5, centerOfBox.getX(), centerOfBox.getX(),
        //                               centerOfBox.getX()};

        //            if(myRank == 0)
        //            {
        //                ierr = MPI_File_write_at(_mpiFile, _headerSize, &boxSim[0], 4, mpiFReal_t, MPI_STATUS_IGNORE);
        //            }
        //            if(ierr > 0)
        //            {
        //                std::cerr << "Error during the construction of the header in "
        //                             "FMpiFmaGenericWriter::writeDistributionOfParticlesFromOctree"
        //                          << std::endl;
        //            }
        //            MPI_Type_size(mpiFReal_t, &sizeType);
        //            _headerSize += sizeType * 4;
        //            //
        //            // Construct the local number of particles on my process
        //            std::size_t maxPartLeaf = 0, nn = 0;
        //            MortonIndex starIndex = mortonLeafDistribution[2 * myRank],
        //                        endIndex = mortonLeafDistribution[2 * myRank + 1];
        //            //  myOctree.template forEachCellLeaf<typename TreeType::LeafClass_T >(
        //            myOctree.forEachCellLeaf(
        //              [&](typename TreeType::CellClassType* cell, typename TreeType::LeafClass_T* leaf) {
        //                  if(!(cell->getMortonIndex() < starIndex || cell->getMortonIndex() > endIndex))
        //                  {
        //                      auto n = leaf->getSrc()->getNbParticles();
        //                      maxPartLeaf = std::max(maxPartLeaf, n);
        //                      nn += n;
        //                  }
        //              });
        //            std::cout << "  nn " << nn << "  should be " << nbLocalParticles << std::endl;
        //            std::vector<FReal> particles(maxPartLeaf * _nbDataTowritePerRecord);
        //            // Build the offset for eaxh processes
        //            std::size_t before = 0;   // Number of particles before me (rank < myrank)
        //            MPI_Scan(const_cast<std::size_t*>(&nbLocalParticles), &before, 1, mpistd::size_t_t, MPI_SUM,
        //                     m_parallelManager->global().getComm());
        //            before -= nbLocalParticles;
        //            MPI_Offset offset = _headerSize + sizeType * _nbDataTowritePerRecord * before;
        //            //
        //            // Write particles in file
        //            myOctree.forEachCellLeaf(
        //              [&](typename TreeType::CellClassType* cell, typename TreeType::LeafClass_T* leaf) {
        //                  if(!(cell->getMortonIndex() < starIndex || cell->getMortonIndex() > endIndex))
        //                  {
        //                      const std::size_t nbPartsInLeaf = leaf->getTargets()->getNbParticles();
        //                      const FReal* const posX = leaf->getTargets()->getPositions()[0];
        //                      const FReal* const posY = leaf->getTargets()->getPositions()[1];
        //                      const FReal* const posZ = leaf->getTargets()->getPositions()[2];
        //                      const FReal* const physicalValues = leaf->getTargets()->getPhysicalValues();
        //                      const FReal* const forceX = leaf->getTargets()->getForcesX();
        //                      const FReal* const forceY = leaf->getTargets()->getForcesY();
        //                      const FReal* const forceZ = leaf->getTargets()->getForcesZ();
        //                      const FReal* const potential = leaf->getTargets()->getPotentials();
        //                      for(int i = 0, k = 0; i < nbPartsInLeaf; ++i, k += _nbDataTowritePerRecord)
        //                      {
        //                          particles[k] = posX[i];
        //                          particles[k + 1] = posY[i];
        //                          particles[k + 2] = posZ[i];
        //                          particles[k + 3] = physicalValues[i];
        //                          particles[k + 4] = potential[i];
        //                          particles[k + 5] = forceX[i];
        //                          particles[k + 6] = forceY[i];
        //                          particles[k + 7] = forceZ[i];
        //                      }
        //                      MPI_File_write_at(_mpiFile, offset, particles.data(),
        //                                        static_cast<int>(_nbDataTowritePerRecord * nbPartsInLeaf), mpiFReal_t,
        //                                        MPI_STATUS_IGNORE);
        //                      offset += sizeType * _nbDataTowritePerRecord * nbPartsInLeaf;
        //                  }
        //              });
        //#ifdef TODO
        //#endif
        //            MPI_File_close(&_mpiFile);
        //        }
    };
}   // namespace scalfmm::io
#endif   // FMPIFMAGENERICLOADER_HPP
