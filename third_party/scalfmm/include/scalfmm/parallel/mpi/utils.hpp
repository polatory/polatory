#ifndef _PARALLEL_MPI_UTILS_HPP_
#define _PARALLEL_MPI_UTILS_HPP_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <cpp_tools/colors/colorized.hpp>
#include <cpp_tools/parallel_manager/parallel_manager.hpp>

#ifdef SCALFMM_USE_MPI
#include "scalfmm/parallel/utils.hpp"

#include <inria/algorithm/distributed/distribute.hpp>
#include <inria/algorithm/distributed/mpi.hpp>
#include <inria/linear_tree/balance_tree.hpp>
#endif
namespace scalfmm::parallel::utils
{

    /**
     * @brief  print the distribution of components (cells/leaves) in a stream
     *
     * @tparam Vector
     * @param out the stream
     * @param header the header to write
     * @param distrib
     */
    template<typename Vector>
    auto inline print_distrib(std::ostream& out, std::string const& header, Vector const& distrib) -> void
    {
        out << header;
        for(auto p: distrib)
        {
            out << '[' << p[0] << ", " << p[1] << "[ ";
        }
        out << std::endl;
    }
    /**
     * @brief print the distribution of components (cells/leaves)
     *
     * @tparam Vector
     * @param header the header to write
     * @param rank  the process id
     * @param distrib  the vector of distribution
     */
    template<typename Vector>
    auto inline print_distrib(std::string const& header, int rank, Vector const& distrib) -> void
    {
        std::string new_header("rank(" + std::to_string(rank) + ") " + header);
        print_distrib(std::cout, new_header, distrib);
    }
    /**
     * @brief construct the morton indexes at the parent level
     *
     *  Update the vector of morton indexes at a level to the cell at the parent level.
     *  We delete all duplicated indexes and we resize the vector
     *  The size of the vector changes
     *
     * @tparam dimension  the dimension of the space
     * @tparam VectorMortonIdx  the type of the vector of Morton index
     * @param[inout] leafMortonIdx  the vector of Morton index
     */
    template<int dimension, typename VectorMortonIdx>
    auto inline move_index_to_upper_level(VectorMortonIdx& leafMortonIdx) -> void
    {   // Move leafMortonIdx to level level_shared
        for(auto& p: leafMortonIdx)
        {
            p = p >> dimension;
        }
        auto last = std::unique(leafMortonIdx.begin(), leafMortonIdx.end());
        leafMortonIdx.erase(last, leafMortonIdx.end());
    }

    ///
    /// \brief send_get_min_morton_idx send Morton index to the left and get value from the right
    ///
    /// \param[in] conf  the mpi conf
    /// \param[in] morton_idx the Morton index to send o send to processor p-1
    /// \return the Morton index coming from the right
    ///
    template<typename index_type>
    [[nodiscard]] index_type send_get_min_morton_idx(cpp_tools::parallel_manager::parallel_manager& para,
                                                     index_type& morton_idx)
    {
        // Setting parameter
        index_type buff_recev{0};
#ifdef SCALFMM_USE_MPI
        auto comm = para.get_communicator();
        int nb_proc = comm.size();
        int my_rank = comm.rank();
        // compute the buffer size
        if(nb_proc != 1)
        {
            cpp_tools::parallel_manager::mpi::request tab_mpi_status;
            auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<index_type>();

            const int sender = (my_rank + 1 == nb_proc) ? MPI_PROC_NULL : my_rank + 1;
            const int receiver = (my_rank == 0) ? MPI_PROC_NULL : my_rank - 1;
            comm.isend(&morton_idx, 1, mpi_type, receiver, 1);

            tab_mpi_status = comm.irecv(&buff_recev, 1, mpi_type, sender, 1);

            // Waiting for the result of the request, if my rank is 0
            // I don't need to wait
            if(my_rank != nb_proc - 1)
            {
                cpp_tools::parallel_manager::mpi::request::waitall(1, &tab_mpi_status);
            }
        }
#endif
        return buff_recev;
    }
#ifdef SCALFMM_USE_MPI

    ///
    /// \brief exchange_data_left_right to exchange data left and right between processor left and right
    ///
    /// The processor p send data_left to processor p-1 and receive from it data_right and
    ///  p send data_right to processor p+1 and receive from it data_left
    /// \param[in] conf
    /// \param[in] data_left data to send to processor left
    /// \param[in] data_right  data to send to processor right
    ///
    /// \return a tuple containing the value_right of processor on the left and the
    ///   value left coming from processor right
    ///
    template<typename data_type>
    auto exchange_data_left_right(cpp_tools::parallel_manager::mpi_config& conf, data_type& data_left,
                                  data_type& data_right)
    {
        // Setting parameter
        data_type buff_p{0}, buff_n{0};
        auto comm = conf.comm;
        int nb_proc = comm.size();
        int my_rank = comm.rank();
        // compute the buffer size
        if(nb_proc != 1)
        {
            // First exchange to the left
            cpp_tools::parallel_manager::mpi::request tab_mpi_status[2];
            //  auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<index_type>();
            // if i'm not the last proc
            const int right = (my_rank + 1 == nb_proc) ? MPI_PROC_NULL : my_rank + 1;
            const int left = (my_rank == 0) ? MPI_PROC_NULL : my_rank - 1;
            const auto length = sizeof(data_left);
            comm.isend(&data_left, length, MPI_CHAR, left, 1);
            comm.isend(&data_right, length, MPI_CHAR, right, 2);

            tab_mpi_status[0] = comm.irecv(&buff_n, length, MPI_CHAR, right, 1);
            tab_mpi_status[1] = comm.irecv(&buff_p, length, MPI_CHAR, left, 2);

            // Waiting for the result of the request if my rank is 0
            // I don't need to wait

            cpp_tools::parallel_manager::mpi::request::waitall(2, tab_mpi_status);

            //////////////
        }
        return std::make_tuple(buff_p, buff_n);
    }
#endif

    ///
    /// \brief Distribute uniformly on the processes the leaves.
    ///
    ///  Split in interval (semi-open) the leaves
    /// The algorithm is
    ///  1) we distribute the particles according to their Morton index. The
    ///  the leaves are split on the processor by their Morton index
    ///
    ///  2) balanced the Morton index by some criteria to define
    ///
    /// parameter[inout] mortonArray Morton index located on the processor
    /// parameter[out] the distribution of leaves

    template<typename MortonArray_type>
    auto balanced_leaves(cpp_tools::parallel_manager::parallel_manager& manager, MortonArray_type& mortonArray)
    {
        //         std::cout << cpp_tools::colors::green << " --> Begin distrib::balanced_leaves  " << cpp_tools::colors::reset
        //                   << std::endl;
        //
        using morton_type = typename MortonArray_type::value_type;

        auto rank = manager.get_communicator().rank();
        // io::print("rank(" + std::to_string(rank) + ") leafMortonIdx: ", mortonArray);
        // auto last = std::unique(mortonArray.begin(), mortonArray.end());
        // mortonArray.erase(last, mortonArray.end());
        //  io::print("rank(" + std::to_string(rank) + ") leafMortonIdx U: ", mortonArray);
        //
        // get max and min of the Morton index owned by current  process
        // [min, max] On two consecutive processes we may have max[p] = min[p+1]
        // we remove such case
        auto minIndex{mortonArray[0]};
        auto maxIndex{mortonArray[mortonArray.size() - 1]};
        const auto nb_proc = manager.get_num_processes();
        std::vector<std::array<morton_type, 2>> cell_distrib(nb_proc, {0, 0});
        if(nb_proc == 1)
        {
            cell_distrib[0] = {minIndex, maxIndex + 1};
        }
#ifdef SCALFMM_USE_MPI
        auto minIndexNextProc = send_get_min_morton_idx(manager, minIndex);

        if(maxIndex == minIndexNextProc)
        {
            mortonArray.resize(mortonArray.size() - 1);
        }
        // Now the Morton indexes are unique on all processes
        //////////////////////////////////////////////////////////////////
        ///  Construct a  uniform distribution of the Morton index
        ///

        MortonArray_type morton_distrib;
        try
        {
            inria::mpi_config conf_tmp(manager.get_communicator().raw_comm);
            inria::distribute(conf_tmp, begin(mortonArray), end(mortonArray), morton_distrib,
                              inria::uniform_distribution{conf_tmp, mortonArray});
        }
        catch(std::out_of_range& e)
        {
            std::cerr << e.what() << '\n';
        }
        //    print("rank(" + std::to_string(rank) + "morton_distrib ", morton_distrib);
        //    manager.comm.barrier();

        //            std::cout << "rank(" + std::to_string(rank) + ") Morton distrib  [" << morton_distrib[0] << ",
        //            "
        //                      << morton_distrib[morton_distrib.size() - 1] << "]\n";

        //print("rank(" + std::to_string(rank) + ") Distrib cells Index: ", morton_distrib);

        cell_distrib.resize(manager.get_num_processes(), {0, 0});
        std::array<morton_type, 2> local{morton_distrib[0], morton_distrib[morton_distrib.size() - 1]};
        cell_distrib[0] = local;
        //      print("rank(" + std::to_string(rank) + ") local: ", local);
        /// share the distribution on all processors
        manager.get_communicator().allgather(local.data(), sizeof(local), MPI_CHAR, cell_distrib.data(), sizeof(local),
                                             MPI_CHAR /*, 0*/);

#endif
        //        std::cout << cpp_tools::colors::red;
        //         io::print("rank(" + std::to_string(rank) + ") cell_distrib: ", cell_distrib);

        //         std::cout << cpp_tools::colors::green << " --> End distrib::balanced_leaves  " << cpp_tools::colors::reset
        //                   << std::endl;
        return cell_distrib;
    }
    ///
    /// \brief balanced_particles compute a balanced particle distribution
    ///
    ///  1) we distribute the particles according to their Morton index. The
    ///  the leaves
    ///   are split on the processor by their Morton index
    ///  2) balanced the Morton index by some criteria to define
    ///
    /// input[in] tha parallel manager
    /// input[in] partArray the vector of particles own by the processor
    /// input[inout] mortonArray the morton index located on the processor
    /// input[in] number_of_particles the total number of particles on all processes
    ///
    /// \return the distribution in terms of Morton index (std::vector<std::array<Morton_type,2>>)
    template<typename ParticleArray_type, typename MortonArray_type>
    auto balanced_particles(cpp_tools::parallel_manager::parallel_manager& manager, ParticleArray_type& partArray,
                            const MortonArray_type& morton_array, const std::size_t& number_of_particles)
    {
        std::cout << cpp_tools::colors::green << " --> Begin distrib::balanced_particles  " << cpp_tools::colors::reset
                  << std::endl;

        using Morton_type = typename MortonArray_type::value_type;
        using MortonDistrib_type = std::array<int, 2>;

        auto rank = manager.get_process_id();
        auto nb_proc = manager.get_num_processes();

        std::vector<std::array<Morton_type, 2>> morton_distrib(nb_proc, {0, 0});
        auto bad_index = static_cast<int>(morton_array.back()) + 1;

        auto LeafMortonIndex(morton_array);
        auto last = std::unique(LeafMortonIndex.begin(), LeafMortonIndex.end());
        LeafMortonIndex.erase(last, LeafMortonIndex.end());
        io::print(" LeafMortonIndex ", LeafMortonIndex);
        /// LeafMortonIndex has the size of the number of leaves
        /// weight = ({Morton index, number of particles}) for each leaf
        std::vector<MortonDistrib_type> weight(LeafMortonIndex.size(), {bad_index, 0});
        std::size_t pos = 0;
        weight[pos][0] = LeafMortonIndex[pos];
        // std::cout << cpp_tools::colors::red << "leaf size: " << LeafMortonIndex.size() << std::endl;
        {   // loop on the number of particles
            for(std::size_t part = 0; part < morton_array.size(); ++part)
            {
                //                    std::cout << "part " << part << " " <<
                //                    tmp[part] << " pos " << pos << " " <<
                //                    leafMortonIdx[pos]
                //                              << "  " << weight[pos] <<
                //                              std::endl;
                while(morton_array[part] != LeafMortonIndex[pos])
                {
                    //                       std::cout << "  new pos " << pos <<
                    //                       std::endl;
                    pos++;
                }
                weight[pos][1] += 1;
                weight[pos][0] = LeafMortonIndex[pos];
            }
            io::print("rank(" + std::to_string(rank) + ") weight: ", weight);
        }

        // get max and min of the Morton index owned by current process
        // [min, max] On two consecutive processes we may have  max[p] = min[p+1]
        //            // we remove such case
        /// Find the minimal and maximal index for the current process
        ///   this index should be unique (check with the neighbors left and right)
        auto minIndex{weight[0]};
        auto maxIndex{weight[weight.size() - 1]};
        if(nb_proc == 1)
        {
            morton_distrib[0] = {minIndex[0], maxIndex[0]};
            // parallel::utils::print_distrib(std::cout, "distrib proc=1", morton_distrib);

            return morton_distrib;
        }
        io::print("rank(" + std::to_string(rank) + ") weight initial: ", weight);

#ifdef SCALFMM_USE_MPI

        cpp_tools::parallel_manager::mpi_config conf(manager.get_communicator());

        MortonDistrib_type weight_prev, weight_next;
        std::tie(weight_prev, weight_next) = exchange_data_left_right(conf, minIndex, maxIndex);

        if(maxIndex[0] == weight_next[0])
        {
            weight.resize(weight.size() - 1);
        }
        if(weight[0][0] == weight_prev[0])
        {
            weight[0][1] += weight_prev[1];
        }

        // io::print("rank(" + std::to_string(rank) + ") weight final: ", weight);
        ///
        /// compute the number of particles in the leaves
        int nb_part = 0;
        for(int i = 0; i < weight.size(); ++i)
        {
            nb_part += weight[i][1];
        }
        // Now the Morton indexes are unique on all processes
        //////////////////////////////////////////////////////////////////
        ///  Construct a  uniform distribution of the Morton index
        ///
        int block = number_of_particles / nb_proc;
        if(nb_proc - 1 == rank)
        {
            block = number_of_particles - rank * block;
        }
        std::cout << "rank(" << rank << ") N particles: " << nb_part << " block " << block << std::endl;

        std::array<Morton_type, 3> local{weight[0][0], weight[weight.size() - 1][0], nb_part};
        std::vector<std::array<Morton_type, 3>> part_distrib(nb_proc);

        part_distrib[0] = local;

        //        io::print("rank(" + std::to_string(rank) + ") 0 Distrib cells Index: ", part_distrib);
        //  std::cout << "rank(" << rank << ") local: " <<local[0]<<" " <<local[1]<<" " <<local[2] <<std::endl;

        /// share the distribution on all processors
        auto nb_elt = sizeof(local);
        conf.comm.allgather(local.data(), nb_elt, MPI_CHAR, part_distrib[0].data(), nb_elt, MPI_CHAR /*, 0*/);
        // io::print("rank(" + std::to_string(rank) + ") Distrib cells Index: ", part_distrib);
        ///
        /// Try to have the same number of particles on a processor
        ///
        block = number_of_particles / nb_proc;
        std::vector<int> tomove(nb_proc, 0);
        std::vector<int> tosendL(nb_proc, 0), tosendR(nb_proc, 0);
        std::vector<int> toreceiv(nb_proc, 0);
        std::vector<Morton_type> numberLeaves(nb_proc, 0);
        Morton_type maxLeaves{0};   // minLeaves{0};
        for(int i = 0; i < nb_proc; ++i)
        {
            /// Prevent to have 0 cell on a processor.
            numberLeaves[i] = part_distrib[i][1] - part_distrib[i][0] + 1;
            maxLeaves = std::max(numberLeaves[i], maxLeaves);
        }
        // io::print("rank(" + std::to_string(rank) + ") numberLeaves: ", numberLeaves);

        // std::cout << "rank(" + std::to_string(rank) + ") initial tomove: " << maxLeaves << std::endl;
        /// Prevent to have 0 cell on a processor.
        if(maxLeaves > 1)
        {
            for(int i = 0; i < nb_proc; ++i)
            {
                tomove[i] = part_distrib[i][2] - block;
            }
        }
        // io::print("rank(" + std::to_string(rank) + ") initial tomove: ", tomove);
        //        if(rank == 0)

        for(int i = 0; i < nb_proc - 1; ++i)
        {
            if(tomove[i] < 0)
            {
                tosendL[i + 1] = -tomove[i];
                tomove[i + 1] += tomove[i];
                tomove[i] = 0;
            }
            else if(tomove[i] > 0)
            {
                tosendR[i + 1] = tomove[i];
                tomove[i] = 0;
                tomove[i + 1] += tomove[i];
            }
            //                    print("   end (" + std::to_string(i) + ")
            //                    tomove: ", tomove); print("   end (" +
            //                    std::to_string(i) + ") tosendR: ",
            //                    tosendR); print("   end (" +
            //                    std::to_string(i) + ") tosendL: ",
            //                    tosendL);
        }
        tosendR[nb_proc - 1] = 0;

        //                       io::print("rank(" + std::to_string(rank) + ") tomove: ", tomove);
        //                        io::print("rank(" + std::to_string(rank) + ") tosendR: ", tosendR);
        //                        io::print("rank(" + std::to_string(rank) + ") tosendRL: ", tosendL);
        ///
        //            std::cout << "tosendL(" + std::to_string(rank) + "): " << tosendL[rank] << std::endl;
        //            std::cout << "tosendR(" + std::to_string(rank) + "): " << tosendR[rank] << std::endl;
        //            if(rank > 0)
        //                std::cout << "toReceivL(" + std::to_string(rank) + "): " << tosendR[rank - 1] <<
        //                std::endl;
        //            if(rank < nb_proc - 1)
        //                std::cout << "toReceivR(" + std::to_string(rank) + "): " << tosendL[rank + 1] <<
        //                std::endl;
        int toReceivL, toReceivR;

        toReceivL = tosendR[rank - 1] > 0 ? 1 : 0;
        toReceivR = tosendL[rank + 1] > 0 ? 1 : 0;
        /// proceed the communications
        ///  first send the number particles inside leaves to send
        ///
        int nb_leaf_to_left{0}, nb_leaf_to_right{0}, nb_part_to_left{0}, nb_part_to_right{0};
        Morton_type morton_to_left{0}, morton_to_right{0};
        MortonDistrib_type MortonPart_to_left{{0, 0}}, MortonPart_to_right{{0, 0}};
        // std::cout << rank << " Morton [ " << MortonPart_to_left << ", " << MortonPart_to_right << "]" << std::endl;

        if(tosendL[rank] > 0)
        {
            int leaf_idx = 0;
            nb_part_to_left = weight[leaf_idx][1];
            //     std::cout << " tosendL  leaf_idx " << leaf_idx << "  " << nb_part_to_left << std::endl;

            while(nb_part_to_left <= tosendL[rank])
            {
                leaf_idx++;
                nb_part_to_left += weight[leaf_idx][1];
                // std::cout << "   tosendL  new pos " << leaf_idx << "  " << nb_part_to_left << std::endl;
            }

            // nb_leaf_to_left = leaf_idx + 1;
            nb_leaf_to_left = leaf_idx;
            morton_to_left = weight[leaf_idx][0];
            MortonPart_to_left = {weight[leaf_idx][0], nb_leaf_to_left};
            // New starting Morton index for the local distribution
            // local[0] = weight[leaf_idx + 1][0];   // Bug here ?
            local[0] = weight[leaf_idx][0];   // Bug here ?

            // std::cout << rank << " local[0] " << local[0] << std::endl;
            // std::cout << rank << " send  morton_to_left" << morton_to_left << std::endl;
        }

        if(tosendR[rank] > 0)
        {
            int leaf_idx = weight.size() - 1;
            nb_part_to_right = weight[leaf_idx][1];
            //   std::cout << "tosendR  leaf_idx " << leaf_idx << "  " << nb_part_to_right << std::endl;

            while(nb_part_to_right <= tosendL[rank])
            {
                leaf_idx--;
                nb_part_to_right += weight[leaf_idx][1];
                //     std::cout << "   - tosendR  new pos " << leaf_idx << "  " << nb_part_to_right <<
                //     std::endl;
            }
            nb_leaf_to_right = leaf_idx + 1;
            morton_to_right = weight[leaf_idx][0];
            MortonPart_to_right = {weight[leaf_idx][0], nb_leaf_to_left};
            // New starting Morton index for the local distribution
            local[1] = weight[leaf_idx][0];

            // std::cout << rank << " send  " << nb_leaf_to_right << " leaf to right - nb  part " << nb_part_to_right
            //           << "  " << MortonPart_to_right[0] << std::endl;
            // std::cout << rank << "send  morton_to_right " << morton_to_right << std::endl;
        }
        local[3] = 0;
        // std::cout << rank << " local partition [ " << local[0] << ", " << local[1] << "]" << std::endl;

        /// Send the number
        /// send to left and right
        // int nb_elt_from_left{0}, nb_elt_from_right{0};
        // Morton_type min_idx{part_distrib[rank][0]}, max_idx{part_distrib[rank][1]};
        Morton_type morton_from_left{local[0]}, morton_from_right{local[1]};

        /// receive from left right
        //                auto exchange_val = [&manager, &rank,
        //                &nb_proc, &tosendL, &tosendR, &toReceivL,
        //                                     &toReceivR](const auto&
        //                                     nb_part_to_left, const
        //                                     auto& nb_part_to_right,
        //                                                 auto&
        //                                                 nb_elt_from_left,
        //                                                 auto&
        //                                                 nb_elt_from_right)
        {
            // compute the buffer size

            cpp_tools::parallel_manager::mpi::request tab_mpi_status[2];
            auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<decltype(morton_from_left)>();
            // if i'm not the last proc
            const int to_right = (rank + 1 == nb_proc) ? MPI_PROC_NULL : rank + 1;
            const int to_left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
            if(tosendL[rank] > 0)
            {
                conf.comm.isend(&morton_to_left, 1, mpi_type, to_left, 1);
            }
            if(tosendR[rank] > 0)
                conf.comm.isend(&morton_to_right, 1, mpi_type, to_right, 1);
            int idx = 0;
            if(toReceivL > 0)
            {
                tab_mpi_status[idx] = conf.comm.irecv(&morton_from_left, 1, mpi_type, to_left, 1);
                local[0] = morton_from_left;

                idx++;
            }
            if(toReceivR > 0)
            {
                tab_mpi_status[idx] = conf.comm.irecv(&morton_from_right, 1, mpi_type, to_right, 1);
                local[1] = morton_from_right;
            }
            // Waiting for the result of the request, if my rank is 0
            // I don't need to wait

            if(toReceivL + toReceivR > 0)
            {
                cpp_tools::parallel_manager::mpi::request::waitall(toReceivL + toReceivR, tab_mpi_status);
            }

            // std::cout << rank << " Morton Left: " << morton_from_left << "  Morton right: " << morton_from_right
            //           << std::endl;
        }

        //    exchange_val(nb_part_to_left, nb_part_to_right,
        //    nb_elt_from_left, nb_elt_from_right);

        //   std::array<Morton_type, 3> local{weight[0][0], weight[weight.size() - 1][0], nb_part};
        // io::print("rank(" + std::to_string(rank) + ") 00 Distrib cells Index: ", part_distrib);

        std::array<Morton_type, 2> local1 = {std::max(morton_from_left, part_distrib[rank][0]),
                                             std::min(morton_from_right, part_distrib[rank][1])};

        // std::cout << rank << " final local 1 [ " << local1[0] << ", " << local1[1] << "]" << std::endl;

        morton_distrib[0] = local1;

        // print("rank(" + std::to_string(rank) + ") Distrib cells Index: ", part_distrib);
        // std::cout << "rank(" << rank << ") Distrib Leaf Index: " <<
        // nb_part << std::endl;

        /// share the distribution on all processors
        nb_elt = sizeof(local1);
        conf.comm.allgather(local1.data(), nb_elt, MPI_CHAR, morton_distrib[0].data(), nb_elt, MPI_CHAR /*, 0*/);
        // io::print("rank(" + std::to_string(rank) + ") Morton distrib final: ", morton_distrib);

#endif
        std::cout << cpp_tools::colors::green << " --> End distrib::balanced_particles  " << cpp_tools::colors::reset
                  << std::endl;
        return morton_distrib;
    }

    template<typename ParticlesArray_type, typename MortonArray_type, typename MortonDistrib_type>
    auto compute_communications(int my_rank, ParticlesArray_type& particles, const MortonArray_type& morton_array,
                                const MortonDistrib_type& morton_dist)
    {
        // std::cout << cpp_tools::colors::green << " --> Begin distrib::compute_communications  "
        //           << cpp_tools::colors::reset << std::endl;

        using Morton_type = typename MortonArray_type::value_type;
        //            auto between = [](const Morton_type& m, const Morton_type& mmin, const Morton_type& mmax) {
        //                return (mmin <= m) && (m <= mmax);
        //            };
        std::vector<int> message(morton_dist.size());
        std::vector<std::array<int, 2>> details_partL(morton_dist.size(), {0, 0}),
          details_partR(morton_dist.size(), {0, 0});
        auto mortonMin = morton_dist[my_rank][0];
        auto mortonMax = morton_dist[my_rank][1];
        /// Compute on the left
        int pos = 0;
        bool new_start = true;
        // std::string beg("rank(" + std::to_string(my_rank) + ")");
        //    if(my_rank == 2)
        {
            // io::print(beg + " morton_dist: ", morton_dist);

            for(std::size_t i = 0; i < particles.size(); ++i)
            {
                if(morton_array[i] >= mortonMin)
                {
                    break;
                }
                while(!math::between(morton_array[i], morton_dist[pos][0], morton_dist[pos][1]))
                {
                    pos++;
                    new_start = true;
                }
                if(pos == my_rank)
                {
                    break;
                }
                if(new_start)
                {
                    details_partL[pos][0] = i;
                    message[pos] = 1;
                    new_start = false;
                }
                details_partL[pos][1] += 1;
                //                    std::cout << beg << i << "    L  m_i " << morton_array[i] << " min " <<
                //                    mortonMin << "  rank "
                //                              << morton_dist[pos] << " " << std::boolalpha
                //                              << between(morton_array[i], morton_dist[pos][0],
                //                              morton_dist[pos][1]) << "  pos " << pos
                //                              << std::endl;
            }
        }
        /// Compute on the right
        //            print("rank(" + std::to_string(my_rank) + ") message: ", message);
        //            print("rank(" + std::to_string(my_rank) + ") details_part: ", details_partL);
        {
            //              print(beg + " morton_dist (Right): ", morton_dist);
            pos = morton_dist.size() - 1;
            // my_rank + 1;

            for(std::size_t i = particles.size() - 1; i > 0; --i)
            {
                //                    std::cout << beg << i << "   R   m_i " << morton_array[i] << " max " <<
                //                    mortonMax << "  rank "
                //                              << morton_dist[pos] << " " << std::boolalpha
                //                              << between(morton_array[i], morton_dist[pos][0],
                //                              morton_dist[pos][1]) << "  pos " << pos
                //                              << std::endl;
                if(morton_array[i] <= mortonMax)
                {
                    break;
                }
                while(!math::between(morton_array[i], morton_dist[pos][0], morton_dist[pos][1]))
                {
                    pos--;
                    new_start = true;
                }
                if(pos == my_rank)
                {
                    break;
                }
                if(new_start)
                {
                    details_partR[pos][0] = i;
                    message[pos] = 1;
                    new_start = false;
                }
                details_partR[pos][1] += 1;
                //                    std::cout << beg << i << "   R   m_i " << morton_array[i] << " max " <<
                //                    mortonMax << "  rank "
                //                              << morton_dist[pos] << " " << std::boolalpha
                //                              << between(morton_array[i], morton_dist[pos][0],
                //                              morton_dist[pos][1]) << "  pos " << pos
                //                              << std::endl;
            }
            //                print("rank(" + std::to_string(my_rank) + ") message: ", message);
            //                print("rank(" + std::to_string(my_rank) + ") details_part R: ", details_partR);
        }
        //	std::cout << cpp_tools::colors::green << " --> End distrib::compute_communications  "
        //		  << cpp_tools::colors::reset << std::endl;
        return std::make_tuple(message, details_partL, details_partR);
    }
    /**
     * @brief
     *
     * @param manager
     * @param particles
     * @param morton_array
     * @param morton_dist
     * @param box
     * @param leaf_level
     * @param total_num_particles
     */
    template<typename ParticlesArray_type, typename MortonArray_type, typename MortonDistrib_type, typename Box_type>
    void fit_particles_in_distrib(cpp_tools::parallel_manager::parallel_manager& manager,
                                  ParticlesArray_type& particles, const MortonArray_type& morton_array,
                                  const MortonDistrib_type& morton_dist, const Box_type& box, const int& leaf_level,
                                  const int& total_num_particles)
    {
        //     std::cout << cpp_tools::colors::green << " --> Begin distrib::fit_particles_in_distrib  "
        //               << cpp_tools::colors::reset << std::endl;
        int my_rank = manager.get_process_id();
        int nb_proc = manager.get_num_processes();
        // std::cout << "  (" << my_rank << ") size " << particles.size() << " "
        //           << morton_array.size() << std::endl;
#ifdef SCALFMM_USE_MPI
        auto comm = manager.get_communicator();
        //            std::cout << "\n------------- fit_particles_in_distrib -------------" << std::endl;
        // io::print("rank(" + std::to_string(my_rank) + ") morton_array: ", morton_array);
        // get the min and the max morton index of the particles own by the
        // process
        // send the number of communication we will receive
        // auto mortonMin = morton_dist[my_rank][0];
        // auto mortonMax = morton_dist[my_rank][1];

        auto to_comm = std::move(compute_communications(my_rank, particles, morton_array, morton_dist));
        //  std::cout << "  (" << my_rank << ") " <<  std::get<0>(to_comm) << std::endl;
        // Send these numbers
        auto nb_message = std::get<0>(to_comm);
        auto nb_length_left = std::get<1>(to_comm);
        auto nb_length_right = std::get<1>(to_comm);
        std::vector<int> message_to_receiv(nb_proc, 0);

        comm.allreduce(nb_message.data(), message_to_receiv.data(), nb_proc, MPI_INT, MPI_SUM);

        //   print("rank(" + std::to_string(my_rank) + ") final message: ", message_to_receiv);

        //
        //            int nb_message_to_receiv =
        //            message_to_receiv[my_rank];
        int buffer_size_left{0}, buffer_size_right{0};
        int nb_left = my_rank > 0 ? nb_length_left[my_rank - 1][1] : 0;
        int nb_right = my_rank + 1 != nb_proc ? nb_length_right[my_rank + 1][1] : 0;

        cpp_tools::parallel_manager::mpi_config conf(comm);
        std::tie(buffer_size_left, buffer_size_right) = exchange_data_left_right(conf, nb_left, nb_right);
        //            std::cout << "rank(" + std::to_string(my_rank) + ")  nb_left: " << nb_left << std::endl;
        //            std::cout << "rank(" + std::to_string(my_rank) + ")  nb_right: " << nb_right << std::endl;
        //            std::cout << "rank(" + std::to_string(my_rank) + ")  buffer_size_left: " <<
        //            buffer_size_left
        //            << std::endl; std::cout << "rank(" + std::to_string(my_rank) + ")  buffer_size_right: " <<
        //            buffer_size_right << std::endl;
        ///
        /// Send the particles
        /// if nb_left >0 we send a communication on the left
        /// if nb_right >0 we send a communication on the right
        /// if buffer_size_left >0 we receive a communication on the left
        /// if buffer_size_right >0 we receive a communication on the right
        using particle_type = typename ParticlesArray_type::value_type;
        particle_type *buffer_left{nullptr}, *buffer_right{nullptr};
        const int to_right = (my_rank + 1 == nb_proc) ? MPI_PROC_NULL : my_rank + 1;
        const int to_left = (my_rank == 0) ? MPI_PROC_NULL : my_rank - 1;

        if(nb_left > 0)
        {
            //                std::cout << my_rank << " send first part to " << to_left << " nb val= " <<
            //                nb_left << " first p "
            //                          << particles[0] << std::endl;

            conf.comm.isend(particles.data(), nb_left * sizeof(particle_type), MPI_CHAR, to_left, 100);
        }
        if(nb_right > 0)
        {
            int start = particles.size() - nb_right;
            //                std::cout << my_rank << " send last part to " << to_right << " nb val= " <<
            //                nb_right
            //                << " first p "
            //                          << particles[start] << std::endl;
            conf.comm.isend(&(particles[start]), nb_right * sizeof(particle_type), MPI_CHAR, to_right, 100);
        }
        ///
        int nb_commL{(buffer_size_left > 0) ? 1 : 0}, nb_commR{(buffer_size_right > 0) ? 1 : 0};
        std::vector<cpp_tools::parallel_manager::mpi::request> tab_mpi_status;
        //            buffer_right = new particle_type[buffer_size_right];

        if(nb_commL > 0)
        {
            buffer_left = new particle_type[buffer_size_left];
            //                std::cout << my_rank << " post a receiv on left " << to_left << " b " <<
            //                buffer_left
            //                << " size "
            //                          << buffer_size_left << std::endl;

            tab_mpi_status.push_back(
              conf.comm.irecv(buffer_left, buffer_size_left * sizeof(particle_type), MPI_CHAR, to_left, 100));
        }
        if(nb_commR > 0)
        {
            buffer_right = new particle_type[buffer_size_right];

            //                std::cout << my_rank << " post a receiv on right " << to_right << " b " <<
            //                buffer_right << " size "
            //                          << buffer_size_right << " " << std::endl;

            tab_mpi_status.push_back(
              conf.comm.irecv(buffer_right, buffer_size_right * sizeof(particle_type), MPI_CHAR, to_right, 100));
        }
        //
        // Prepare the copy during the communications
        //
        int new_part_size = particles.size() - nb_left - nb_right + buffer_size_left + buffer_size_right;
        //   std::cout << my_rank << " old size " << particles.size() << " new size " << new_part_size <<
        //   std::endl;

        ParticlesArray_type newArray(new_part_size);
        /// Here we copy in the right place the particles that do not move
        auto start = particles.begin() + nb_left /*std::advance(particles.begin(), nb_left)*/;
        auto end = particles.end() - nb_right /* std::advance(std::begin(particles), particles.size() - nb_right)*/;
        auto start_out = newArray.begin() + buffer_size_left /*std::advance(std::begin(newArray), buffer_size_left)*/;
        std::copy(start, end, start_out);

        //       conf.comm.barrier();
        //            std::cout << my_rank << " status size " << tab_mpi_status.size() << std::endl;
        if(tab_mpi_status.size() > 0)
        {
            //                std::cout << my_rank << " I'm waiting  " << tab_mpi_status.size() << " " <<
            //                buffer_left << "  "
            //                          << buffer_right << std::endl;
            for(int j = 0; j < tab_mpi_status.size(); ++j)
            {
                cpp_tools::parallel_manager::mpi::status status;
                tab_mpi_status[j].get_status(status);
                //                    std::cout << my_rank << " request " << j << "  count " <<
                //                    status.get_count(MPI_CHAR) << " source "
                //                              << status.source() << " tag " << status.tag() << std::endl;
            }
            cpp_tools::parallel_manager::mpi::request::waitall(tab_mpi_status.size(), tab_mpi_status.data());
        }
        conf.comm.barrier();

        //            std::cout << my_rank << " ---------- End Redistribution ----------" << std::endl;
        if(buffer_left)
        {
            /// Here we copy in the right place the particles that do not move
            std::copy(buffer_left, buffer_left + buffer_size_left, newArray.begin());

            delete[] buffer_left;
        }
        if(buffer_right)
        {
            /// Here we copy in the right place the particles that do not move
            std::copy(buffer_right, buffer_right + buffer_size_right, newArray.end() - buffer_size_right);

            delete[] buffer_right;
        }
        particles = std::move(newArray);
        ///
        /// Check if we still have th good number of particles
        using int_length = std::int64_t;
        int_length new_num_particles{0};
        int_length local_num_particles{static_cast<int_length>(particles.size())};
        auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<int_length>();
        conf.comm.allreduce(&local_num_particles, &new_num_particles, 1, mpi_type, MPI_SUM);
        //
        if(new_num_particles - total_num_particles != 0)
        {
            std::cerr << " Total number of particles: (new) " << new_num_particles << " (old) " << total_num_particles
                      << std::endl;
            std::cerr << " we lost some particles " << std::endl;
            manager.end();
            std::exit(EXIT_FAILURE);
        }
#else
        new_num_particles = particles.size();
#endif

        // std::cout << cpp_tools::colors::green << " --> End distrib::fit_particles_in_distrib  "
        //           << cpp_tools::colors::reset << std::endl;
    }
    ///
    /// \brief Build the cell distribution at one level upper
    /// \param[in]  para the parallel manager
    /// \param[in] dimension of the problem
    /// \param[inout] mortonCellIndex the index cell at level+1 (in) and we
    ///  construct the parent cells (out)
    /// \param[in] level current level to construct the cell distribution
    /// \param[in] cellDistrib at level + 1
    ///  \return the cell distribution at level
    ///
    template<typename VectorMortonIdx, typename MortonDistribution>
    inline auto build_upper_distribution(cpp_tools::parallel_manager::parallel_manager& para,
                                         const std::size_t dimension, const int& level,
                                         VectorMortonIdx& mortonCellIndex, std::int64_t& ghost_l2l,
                                         const MortonDistribution& cells_distrib) -> MortonDistribution
    {
        using morton_type = typename VectorMortonIdx::value_type;
        // std::cout << cpp_tools::colors::blue << " --> Begin distrib::build_upper_distribution at level " << level
        //           << cpp_tools::colors::reset << std::endl;
        // std::cout << std::endl;
        MortonDistribution parent_distrib(cells_distrib);
        auto rank = para.get_process_id();
        // std::int64_t ghost_parent{-1};
        //            io::print("rank(" + std::to_string(rank) + ") cells_distrib: ", cells_distrib);
        //            io::print("rank(" + std::to_string(rank) + ") mortonCellIndex: ", mortonCellIndex);

        // get the parent distribution
        for(auto& p: parent_distrib)
        {
            p[0] = (p[0] >> dimension);
            p[1] = ((p[1] - 1) >> dimension);   // -1 to have the last morton index inside
                                                // check if I m the first
        }
        ///
        /// Check if there are no shared bounds (i.e. the children of the parent are on two mpi processes)
        /// The distribution is a range (i.e. [a, b[)
        ///
        std::vector<bool> need_comm(std::size_t(para.get_num_processes()), false);
        std::size_t start = 0;
        // io::print("rank(" + std::to_string(rank) + ") parent_distrib: ", parent_distrib);
        for(int p = 1; p < para.get_num_processes(); ++p)
        {
            // std::cout << rank << " p " << p << " " << std::boolalpha << need_comm[p];
            if(parent_distrib[p - 1][1] == parent_distrib[p][0])
            {
                // move to one the parent doest not own the first child
                parent_distrib[p][0] += 1;
                need_comm[p] = true;
            }
            parent_distrib[p - 1][1] += 1;   // to have an semi-open interval  (like range)
            // std::cout << "   -> " << need_comm[p] << std::endl;
            ;
        }
        parent_distrib[para.get_num_processes() - 1][1] += 1;   // to have a semi-open interval  (like range)
        // print_distrib("parent_distrib(add):", rank, parent_distrib);

        // morton_type m2m_ghost_parent;
        VectorMortonIdx send_to_left;
        std::size_t max_children = pow(2, dimension);
        if(need_comm[rank])
        {
            // std::cout << cpp_tools::colors::red << "Need to adjust the distribution (not checked) on rank " << rank
            //           << " \n";
            // print_distrib("parent_distrib(good):", rank, parent_distrib);

            // Check if we need communication in upward and downward pass
            start = 1;
            // io::print("Distrib", parent_distrib[rank]);
            // The parent may be on the previous process not on mine and -1 because distrib is a range
            auto last_parent_on_prev_proc = parent_distrib[rank - 1][1] - 1;
            for(std::size_t i = 0; i < max_children; ++i)
            {
                morton_type parent = (mortonCellIndex[i] >> dimension);
                // std::cout << i << " m " << mortonCellIndex[i] << "  " << parent;

                if(parent == last_parent_on_prev_proc)
                {
                    morton_type first_child = (parent << dimension);
                    // std::cout << " first_child " << first_child << std::endl;
                    if(first_child == mortonCellIndex[i])
                    {
                        break;
                    }

                    send_to_left.push_back(mortonCellIndex[i]);
                }
                else if(parent >= parent_distrib[rank][0])
                {
                    // std::cout << std::endl;
                    break;
                }
                // std::cout << std::endl;
            }
            if(send_to_left.size() > 0)
            {
                ghost_l2l = (mortonCellIndex[0] >> dimension);
                // std::cout << " Add ghost at parent level (l2l) " << ghost_l2l << std::endl;
            }
            // io::print("rank " + std::to_string(rank) + "ghost to send_to_left", send_to_left);
            // std::cout << " ghost_l2l: " << ghost_l2l << std::endl;

            // std::cout << cpp_tools::colors::reset << '\n';
        }
        ///
        /// Build the new morton cell of the parent
        ///
        for(std::size_t i = 0; i < mortonCellIndex.size(); ++i)
        {
            mortonCellIndex[i] = (mortonCellIndex[i] >> dimension);
        }
        /// we have to remove some elements at the beginning if start != 0
        /// the morton indexes are already sorted
        auto last = std::unique(mortonCellIndex.begin(), mortonCellIndex.end());
        if(start > 0)   // We delete the parent belonging to the previous process
        {
            parent_distrib[rank][0] = mortonCellIndex[1];
            VectorMortonIdx new_cell_index(std::distance(mortonCellIndex.begin() + start, last));
            std::move(mortonCellIndex.begin() + start, last, new_cell_index.begin());
            mortonCellIndex = std::move(new_cell_index);
        }
        else
        {
            mortonCellIndex.erase(last, mortonCellIndex.end());
        }

        // io::print("rank(" + std::to_string(rank) + ") mortonCellIndex1: ", mortonCellIndex);
        parent_distrib[0][0] = parent_distrib[rank][0];
        parent_distrib[0][1] = parent_distrib[rank][1];
        auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<morton_type>();

        /// share the distribution on all processors
        para.get_communicator().allgather(parent_distrib.data(), 2, mpi_type);

        // print_distrib("parent_distrib(allgather):", rank, parent_distrib);

        // std::cout << cpp_tools::colors::blue << " --> End distrib::build_upper_distribution at level " << level
        //           << cpp_tools::colors::reset << std::endl;
        return parent_distrib;
    }

    ///
    /// \brief merge two sorted vectors
    ///
    /// Elements appear only once
    ///
    ///  \param[in] v1 first vector to merge
    ///  \param[in] v2 vector to merge
    ///
    /// \return the merged vector
    /// to the first vector
    ///
    template<typename VectorMortonIdx>
    inline VectorMortonIdx merge_unique(VectorMortonIdx& v1, const VectorMortonIdx& v2)
    {
        /*            std::cout << cpp_tools::colors::green << " --> Begin let::merge_unique  " <<
           cpp_tools::colors::reset
                              << std::endl*/
        VectorMortonIdx dst;
        std::merge(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(dst));
        auto last = std::unique(dst.begin(), dst.end());
        //	std::cout << "  last "  << *last <<std::endl;
        dst.erase(last, dst.end());
        //            std::cout << cpp_tools::colors::green << " --> End let::merge_unique  " <<
        //            cpp_tools::colors::reset
        //                      << std::endl;
        //	io::print(" merge uniq dst", dst);
        return dst;
    }

    template<typename VectorMortonIdx, typename Iterator>
    VectorMortonIdx merge_unique_fast(const Iterator a_beg, const Iterator a_end, const Iterator b_beg,
                                      const Iterator b_end)
    {
        // io::print(std::cout, " merge uniq v1", a_beg, a_end);
        // std::cout << std::endl;
        // io::print(std::cout, " merge uniq v2", b_beg, b_end);
        // std::cout << std::endl;
        int n = std::distance(b_beg, b_end);
        int m = std::distance(a_beg, a_end);

        if(m == 0)
        {
            VectorMortonIdx merged(n);
            std::copy(b_beg, b_end, merged.begin());
            return merged;
        }
        if(n == 0)
        {
            VectorMortonIdx merged(m);
            std::copy(a_beg, a_end, merged.begin());
            return merged;
        }
        int j{0};
        std::vector<int> add(n);
        int nb_elt_to_add{0};
        Iterator it_a{a_beg}, it_b{b_beg};
        while(it_b != b_end)
        {
            if(*it_b < *it_a)
            {
                add[j] = 1;
                ++nb_elt_to_add;
                ++j;
                ++it_b;
            }
            else if(*it_b > *it_a)
            {
                ++it_a;
            }
            else
            {
                // same element
                add[j] = 0;
                ++it_b;
                ++it_a;
            }
        }
        // n = std::distance(a_beg, a_end);

        VectorMortonIdx merged(m + nb_elt_to_add, -1);

        it_a = a_beg;
        it_b = b_beg;
        auto it = std::begin(merged);
        while(it_b != b_end)
        {
            if(*it_b < *it_a)
            {
                *it = *it_b;
                ++it_b;
                ++it;
            }
            else if(*it_b > *it_a)
            {
                *it = *it_a;
                ++it_a;
                ++it;
            }
            else
            {
                // same element
                *it = *it_a;
                ++it_b;
                ++it_a;
                ++it;
            }
        }
        std::copy(it_a, a_end, it);
        // io::print("merged ", merged);
        return merged;
    }

    ///
    /// \brief find if the index exists owning the index
    /// \param[in] index
    /// \param[in] distrib the index distribution
    /// \param[in] start [optional] position to start in the distribution
    /// vector \return the process number (if -1 index not in my distribution)
    ///
    template<typename MortonIdx, typename VectorMortonIdx>
    inline std::int64_t find_index(const MortonIdx& index, const VectorMortonIdx& my_index, std::size_t& start)
    {
        for(std::size_t i = start; i < my_index.size(); ++i)
        {
            // std::cout << " <-- " << index << " < ? " << my_index[i] << "--> ";
            if(index > my_index[i])
            {
                continue;
            }
            else if(index == my_index[i])
            {
                start = i + 1;
                return i;
            }
            else if(index < my_index[i])
            {
                start = i;
                return -1;
            }
        }
        return -1;
    }
    template<typename MortonIdx, typename LeafInfo>
    inline std::int64_t find_index2(const MortonIdx& index, const LeafInfo& my_leaves, std::size_t& start)
    {
        for(std::size_t i = start; i < my_leaves.size(); ++i)
        {
            // std::cout << " <-- " << index << " < ? " << my_index[i] << "--> ";
            if(index > my_leaves[i].morton)
            {
                continue;
            }
            else if(index == my_leaves[i].morton)
            {
                start = i + 1;
                return i;
            }
            else if(index < my_leaves[i].morton)
            {
                start = i;
                return -1;
            }
        }
        return -1;
    }
    ///
    /// \brief check if the morton index used in the vector of indexes exist
    ///
    ///  This step needs communication
    /// \param para the parallel manager
    /// \param needed_idx the index to check if they exits in the other processors
    /// \param distrib the index distribution on all processors
    /// \param local_morton_idx My local morton index
    ///
    template<typename VectorMortonIdx, typename MortonDistribution>
    void check_if_morton_index_exist(cpp_tools::parallel_manager::parallel_manager& para, VectorMortonIdx& needed_idx,
                                     const MortonDistribution& distrib, const VectorMortonIdx& local_morton_idx)
    {
        auto rank = para.get_process_id();
        auto nb_proc = para.get_num_processes();

        // std::cout << cpp_tools::colors::green << " (" << rank << ")--> Begin distrib::check_if_morton_index_exist
        // "
        //           << cpp_tools::colors::reset << std::endl;
        // bad_index the largest morton index in the whole indexes + 1
        auto bad_index = distrib[nb_proc - 1][1] + 1;
#ifdef SCALFMM_USE_MPI
        using mortonIdx_type = typename VectorMortonIdx::value_type;

        std::vector<int> nb_messages_to_send(nb_proc, 0);
        std::vector<int> nb_messages_to_receive(nb_proc, 0);
        // begin index to send to process k
        std::vector<int> start(nb_proc + 1, 0);
        //
        // Check the number of messages to send to the processes = number of morton index inside the
        // interval of the distribution
        int k = 0;
        // int start_mine = 0;
        start[nb_proc] = needed_idx.size();
        for(std::size_t i = 0; i < needed_idx.size(); ++i)
        {
            if(needed_idx[i] < distrib[k][1])
            {
                ++nb_messages_to_send[k];
            }
            else
            {
                // find the new processor owing the index
                k = parallel::utils::find_proc_for_index(needed_idx[i], distrib, k);
                if(k > 0)
                {
                    start[k] = i;
                    ++nb_messages_to_send[k];
                }
            }
        }
        start[rank] = start[rank + 1];

        // exchange the vector all to all to know which process send us a
        // message
        // io::print("rank(" + std::to_string(rank) + ") Nb msg send : ", nb_messages_to_send);

        auto comm = para.get_communicator();
        auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<int>();
        comm.alltoall(nb_messages_to_send.data(), 1, mpi_type, nb_messages_to_receive.data(), 1, mpi_type);
        // io::print("rank(" + std::to_string(rank) + ") Nb msg receiv : ", nb_messages_to_receive);
        //
        // check if the morton index exist locally
        // bad_index the largest morton index in the whole indexes + 1
        std::vector<cpp_tools::parallel_manager::mpi::request> tab_mpi_status;
        std::vector<mortonIdx_type*> buffer(nb_proc, nullptr);
        mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<mortonIdx_type>();
        for(std::size_t i = 0; i < nb_messages_to_send.size(); ++i)
        {
            // send the idex tp proc i
            if(nb_messages_to_send[i] != 0)
            {
                comm.isend(&(needed_idx[start[i]]), nb_messages_to_send[i], mpi_type, i, 200);
            }
        }
        for(std::size_t i = 0; i < nb_messages_to_receive.size(); ++i)
        {
            if(nb_messages_to_receive[i] != 0)
            {
                buffer[i] = new mortonIdx_type[nb_messages_to_receive[i]];
                //                    std::cout << "Receive message from " << i << " of size " <<
                //                    nb_messages_to_receive[i] << std::endl;
                tab_mpi_status.push_back(comm.irecv(buffer[i], nb_messages_to_receive[i], mpi_type, i, 200));
            }
        }
        if(tab_mpi_status.size() > 0)
        {
            cpp_tools::parallel_manager::mpi::request::waitall(tab_mpi_status.size(), tab_mpi_status.data());
        }
        ///  Check if the index exist
        ///
        for(std::size_t i = 0; i < nb_messages_to_receive.size(); ++i)
        {
            if(nb_messages_to_receive[i] != 0)
            {
                {
                    auto& elt = buffer[i];
                    //                        std::cout << "rank(" << rank << ")
                    //                        newbuffer[" << i << "] "; for(int ki
                    //                        = 0; ki < nb_messages_to_receive[i];
                    //                        ++ki)
                    //                        {
                    //                            std::cout << elt[ki] << " ";
                    //                        }
                    //  std::cout << std::endl;
                    std::size_t start = 0;
                    for(int ki = 0; ki < nb_messages_to_receive[i]; ++ki)
                    {
                        // std::cout << ki << " " << elt[ki] << "  " << start << " ";
                        //  check if morton index exists in my local morton vector
                        //  if not we must insert a the bad_index in order not to change the size of the buffer to
                        //  send.
                        auto pos = find_index(elt[ki], local_morton_idx, start);
                        if(pos < 0)
                        {
                            elt[ki] = bad_index;
                        }
                        // std::cout << "pos " << pos << " elt  " << elt[ki] << "  " << local_morton_idx[pos]
                        //           << std::endl;
                    }
                    //                        std::cout << "rank(" << rank << ")
                    //                        newbuffer[" << i << "] "; for(int ki
                    //                        = 0; ki < nb_messages_to_receive[i];
                    //                        ++ki)
                    //                        {
                    //                            std::cout << elt[ki] << " ";
                    //                        }
                    // std::cout << std::endl;
                }
            }
        }

        /////////////////////////////////////////////////////////////////////////////
        ///    re-send the existing index
        for(std::size_t i = 0; i < nb_messages_to_send.size(); ++i)
        {
            if(nb_messages_to_receive[i] != 0)
            {
                //                    std::cout << "rank(" << rank << ")  buffer[" << i << "] of size " <<
                //                    nb_messages_to_receive[i]
                //                              << " to proc " << i << std::endl
                //                              << std::flush;

                comm.isend(buffer[i], nb_messages_to_receive[i], mpi_type, i, 300);
            }
        }

        tab_mpi_status.resize(0);
        for(std::size_t i = 0; i < nb_messages_to_send.size(); ++i)
        {
            if(nb_messages_to_send[i] != 0)
            {
                //                    std::cout << "rank(" << rank << ")  Receive message from " << i << " of
                //                    size "
                //                              << nb_messages_to_send[i] << " located at " << start[i] <<
                //                              std::endl
                //                              << std::flush;
                tab_mpi_status.push_back(comm.irecv(&(needed_idx[start[i]]), nb_messages_to_send[i], mpi_type, i, 300));
            }
        }
        if(tab_mpi_status.size() > 0)
        {
            cpp_tools::parallel_manager::mpi::request::waitall(tab_mpi_status.size(), tab_mpi_status.data());
        }
        //                   io::print("rank(" + std::to_string(rank) + ") needed_idx : ", needed_idx);

        for(std::size_t i = 0; i < nb_messages_to_receive.size(); ++i)
        {
            if(nb_messages_to_receive[i] != 0)
            {
                delete[] buffer[i];
            }
        }

#endif
        // We remove the bad_index in order to have only the existing components (leaf/cell)
        std::sort(needed_idx.begin(), needed_idx.end());
        auto last = std::unique(needed_idx.begin(), needed_idx.end());
        //   io::print("rank(" + std::to_string(rank) + ") uniq needed_idx : ", needed_idx);
        if(*(last - 1) == bad_index)
        {
            last = last - 1;
        }
        needed_idx.erase(last, needed_idx.end());
        // // io::print("rank(" + std::to_string(rank) + ") needed_idx : ", needed_idx);
        // std::cout << cpp_tools::colors::green << " (" << rank << ") --> End distrib::check_if_morton_index_exist
        // "
        //           << cpp_tools::colors::reset << std::endl
        //           << std::flush;
    }

    template<typename VectorMortonIdx, typename leafInfo, typename MortonDistribution>
    void check_if_leaf_morton_index_exist(cpp_tools::parallel_manager::parallel_manager& para,
                                          VectorMortonIdx& needed_idx, const MortonDistribution& distrib,
                                          const leafInfo& leaf_info)
    {
        auto rank = para.get_process_id();
        auto nb_proc = para.get_num_processes();

        //         std::cout << cpp_tools::colors::green << " (" << rank
        //                   << ")--> Begin distrib::check_if_leaf_morton_index_exist  " << cpp_tools::colors::reset
        //                  << std::endl;
        //        io::print("rank(" + std::to_string(rank) + ") needed_idx(p2p)  : ", needed_idx);

#ifdef SCALFMM_USE_MPI
        using mortonIdx_type = typename VectorMortonIdx::value_type;

        std::vector<int> nb_messages_to_send(nb_proc, 0);
        std::vector<int> nb_messages_to_receive(nb_proc, 0);
        // begin index to send to process k
        std::vector<int> start(nb_proc + 1, 0);
        //
        // Check the number of messages to send to the processes = number of morton index insize the
        // interval of the distribution
        int k = 0;
        start[nb_proc] = needed_idx.size();
        for(std::size_t i = 0; i < needed_idx.size(); ++i)
        {
            if(needed_idx[i] < distrib[k][1])
            {
                nb_messages_to_send[k]++;
            }
            else
            {
                // find the new processor owing the index
                k = parallel::utils::find_proc_for_index(needed_idx[i], distrib, k);
                if(k >= 0)
                {
                    start[k] = i;
                    nb_messages_to_send[k]++;
                }
            }
        }
        start[rank] = start[rank + 1];
        //         io::print("rank(" + std::to_string(rank) + ") start : ", start);
        //         io::print("rank(" + std::to_string(rank) + ") needed_idx : ", needed_idx);
        //         io::print("rank(" + std::to_string(rank) + ") Nb msg send : ", nb_messages_to_send);

        // exchange the vector all to all to know which process send us a
        // message
        auto comm = para.get_communicator();
        auto mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<int>();
        comm.alltoall(nb_messages_to_send.data(), 1, mpi_type, nb_messages_to_receive.data(), 1, mpi_type);
        //         io::print("rank(" + std::to_string(rank) + ") Nb msg receiv : ", nb_messages_to_receive);
        //  _to_send
        // check if the morton index exist locally
        // bad_index the largest morton index in the whole indexes + 1
        // auto bad_index = distrib[nb_proc - 1][1] + 1;
        std::vector<cpp_tools::parallel_manager::mpi::request> tab_mpi_status;
        std::vector<mortonIdx_type*> buffer(nb_proc, nullptr);
        mpi_type = cpp_tools::parallel_manager::mpi::get_datatype<mortonIdx_type>();
        for(std::size_t i = 0; i < nb_messages_to_send.size(); ++i)
        {
            if(nb_messages_to_send[i] != 0)
            {
                //                    std::cout << "Send message to " << i << " of size " <<
                //                    nb_messages_to_send[i]
                //                    << " located at "
                //                              << start[i] << std::endl;

                comm.isend(&(needed_idx[start[i]]), nb_messages_to_send[i], mpi_type, i, 200);
            }
        }
        for(std::size_t i = 0; i < nb_messages_to_receive.size(); ++i)
        {
            if(nb_messages_to_receive[i] != 0)
            {
                buffer[i] = new mortonIdx_type[nb_messages_to_receive[i]];
                //                    std::cout << "Receive message from " << i << " of size " <<
                //                    nb_messages_to_receive[i] << std::endl;
                tab_mpi_status.push_back(comm.irecv(buffer[i], nb_messages_to_receive[i], mpi_type, i, 200));
            }
        }
        if(tab_mpi_status.size() > 0)
        {
            cpp_tools::parallel_manager::mpi::request::waitall(tab_mpi_status.size(), tab_mpi_status.data());
        }

        ///  Check if the index exist
        ///
        for(std::size_t i = 0; i < nb_messages_to_receive.size(); ++i)
        {
            if(nb_messages_to_receive[i] != 0)
            {
                //  buffer[i];
                //    if(rank == 0)
                {
                    auto& elt = buffer[i];
                    //                        std::cout << "rank(" << rank << ")
                    //                        newbuffer[" << i << "] "; for(int ki
                    //                        = 0; ki < nb_messages_to_receive[i];
                    //                        ++ki)
                    //                        {
                    //                            std::cout << elt[ki] << " ";
                    //                        }
                    //  std::cout << std::endl;
                    std::size_t start = 0;
                    for(int ki = 0; ki < nb_messages_to_receive[i]; ++ki)
                    {
                        //‡ std::cout << ki << " " << elt[ki] << "  " << start << "
                        //";
                        // check if morton index exists in my local morton vector
                        // if not we must insert a the bad_index in order not to change
                        // the size of the buffer to send.
                        auto pos = find_index2(elt[ki], leaf_info, start);
                        if(pos < 0)
                        {
                            elt[ki] = 0;
                        }
                        else
                        {
                            elt[ki] = leaf_info[pos].number_of_particles;
                        }
                    }
                    //                        std::cout << "rank(" << rank << ")
                    //                        newbuffer[" << i << "] "; for(int ki
                    //                        = 0; ki < nb_messages_to_receive[i];
                    //                        ++ki)
                    //                        {
                    //                            std::cout << elt[ki] << " ";
                    //                        }
                    // std::cout << std::endl;
                }
            }
        }
        /////////////////////////////////////////////////////////////////////////////
        ///    re-send the existing index
        for(std::size_t i = 0; i < nb_messages_to_send.size(); ++i)
        {
            if(nb_messages_to_receive[i] != 0)
            {
                //                    std::cout << "rank(" << rank << ")  buffer[" << i << "] of size " <<
                //                    nb_messages_to_receive[i]
                //                              << " to proc " << i << std::endl
                //                              << std::flush;

                comm.isend(buffer[i], nb_messages_to_receive[i], mpi_type, i, 300);
            }
        }

        tab_mpi_status.resize(0);
        for(std::size_t i = 0; i < nb_messages_to_send.size(); ++i)
        {
            if(nb_messages_to_send[i] != 0)
            {
                //                    std::cout << "rank(" << rank << ")  Receive message from " << i << " of
                //                    size "
                //                              << nb_messages_to_send[i] << " located at " << start[i] <<
                //                              std::endl
                //                              << std::flush;
                tab_mpi_status.push_back(comm.irecv(&(needed_idx[start[i]]), nb_messages_to_send[i], mpi_type, i, 300));
            }
        }
        if(tab_mpi_status.size() > 0)
        {
            cpp_tools::parallel_manager::mpi::request::waitall(tab_mpi_status.size(), tab_mpi_status.data());
        }
        // io::print("rank(" + std::to_string(rank) + ") needed_idx : ", needed_idx);

        for(std::size_t i = 0; i < nb_messages_to_receive.size(); ++i)
        {
            if(nb_messages_to_receive[i] != 0)
            {
                delete[] buffer[i];
            }
        }

#endif

        // io::print("rank(" + std::to_string(rank) + ") needed_idx : ", needed_idx);
        // std::cout << cpp_tools::colors::green << " (" << rank
        //           << ") --> End distrib::check_if_leaf_morton_index_exist " << cpp_tools::colors::reset <<
        //           std::endl
        //           << std::flush;
    }

    ///
    /// \brief find the group owning the index
    ///
    /// \param[in] index the index
    /// \param[in] begin iterator to start search
    /// \param[in] end iterator to complete the search
    /// vector \return the process number
    ///
    template<typename MortonIdx, typename Group_iterator_t>
    inline Group_iterator_t find_group_for_index(Group_iterator_t begin, Group_iterator_t end, const MortonIdx& index)
    {
        for(auto grp_ptr = begin; grp_ptr != end; ++grp_ptr)
        {
            auto const& csymb = (*grp_ptr)->csymbolics();

            if(math::between(index, csymb.starting_index, csymb.ending_index))
            {
                return grp_ptr;
            }
        }
        return end;
    }
    /// @brief  Build the vector of dependencies
    ///
    ///       Build the vector of dependencies by adding the first address of the multipoles inside all groups
    ///        between begin and end
    /// @tparam Group_iterator_t
    /// @tparam MortonIdxVector_t
    /// @tparam Dependencies_t
    /// @param begin begin iterator on group
    /// @param end   end iterator on group
    /// @param morton_to_send the Morton index of the cell to send
    /// @param deps the vector of dependencies
    template<typename Group_iterator_t, typename MortonIdxVector_t, typename Dependencies_t>
    void build_multipoles_dependencies_from_morton_vector(Group_iterator_t begin, Group_iterator_t end,
                                                          const MortonIdxVector_t& morton_to_send, Dependencies_t& deps)
    {
        const int max_idx = morton_to_send.size();   // loop on the groups
        // Find the group containing the first index
        auto new_begin = find_group_for_index(begin, end, morton_to_send[0]);
        int idx{0};
        bool found{true};

        for(auto grp_ptr = new_begin; grp_ptr != end; ++grp_ptr)
        {
            auto const& csymb = (*grp_ptr)->csymbolics();
            // iterate on the cells in the same group
            // we move forward in the index vector
            while(idx < max_idx and math::between(morton_to_send[idx], csymb.starting_index, csymb.ending_index))
            {   // find cell inside the group
                found = true;
                ++idx;
            }
            if(found)
            {
                deps.push_back(&(grp_ptr->get()->ccomponent(0).cmultipoles(0)));
                found = false;
            }
        }
    }
    template<typename Group_iterator_t, typename MortonIdxVector_t>
    auto serialise(std::pair<Group_iterator_t, Group_iterator_t> first,
                   std::pair<Group_iterator_t, Group_iterator_t> second, const MortonIdxVector_t& mortons)
    {
    }

}   // namespace scalfmm::parallel::utils

#endif
