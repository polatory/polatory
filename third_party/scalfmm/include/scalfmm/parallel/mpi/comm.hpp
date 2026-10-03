#pragma once

#include <stdexcept>
#include <vector>

#include "scalfmm/container/particle.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/parallel/utils.hpp"

#include <cpp_tools/parallel_manager/parallel_manager.hpp>

#include <mpi.h>

namespace scalfmm::parallel::comm
{
    using comm_type = cpp_tools::parallel_manager::mpi::communicator;

    // Debug functions
    namespace debug
    {
        template<typename TreeType>
        void print_all_cells(TreeType& tree, int level, std::string title)
        {
            std::cout << "M2L  --  level " << level << "   --  " << std::endl;
            scalfmm::component::for_each_mine_component(tree.begin_cells(level), tree.end_cells(level),
                                                        [&title](auto const& cell)
                                                        {
                                                            std::cout << title << " morton " << cell.index()
                                                                      << "  multipoles "
                                                                      << cell.transfer_multipoles().at(0) << "  locals "
                                                                      << cell.locals().at(0) << std::endl
                                                                      << std::flush;
                                                        });
        }
        template<typename VectorAccesType>
        inline void set_values(VectorAccesType& cells_to_send_access)
        {
            std::cout << " debug::set_values " << cells_to_send_access.size() << std::endl << std::flush;
            // iterate on the cells
            int start{20};
            for(auto access: cells_to_send_access)
            {
                auto& cell = (*(access.first))->component(access.second);

                //			     std::cout << " morton to find " << index_to_send[idx] << " cell found "
                //			               << (*grp_ptr)->component(pos).csymbolics().morton_index << '\n';
                auto& m = cell.transfer_multipoles();
                auto nb_m = m.size();
                //			    std::cout << "          nb_m" <<  m.size() <<std::endl;
                for(std::size_t i{0}; i < nb_m; ++i)
                {
                    auto& ten = m.at(i);
                    // std::copy(std::begin(ten), std::end(ten), it);
                    ten[0] = start;
                    std::cout << " start " << start << "   " << ten[0] << std::endl;
                    ++start;
                }
            }
        }
    }   // namespace debug

    /**
     * @brief Determines the Morton index vector to be received from processor p  (STEP 1)
     * 
     * Determines the Morton index vector to be received from processor p. In addition, for each Morton index we
    * store the cell, i.e. a pointer to its group and the index within the group (group_ptr, index). This will
    * enable us to build the MPi data type the multipoles received from processor p directly into the cell.
    *
    * leaf_to_receive_access: a vector of vector of pair (the iterator on the group ,the position of the cell in the group)
    *  leaf_to_receive_access[p] is the position vector of cells in groups whose Morton index comes from processor p
    *  leaf_to_receive_access[p][i] a pair (the iterator on the group ,the position of the cell in the group)
    * vector of size nb_proc
    *    - nb_messages_to_receive: the number of morton indices to exchange with processor p
    *    - nb_messages_to_send: the number of morton indices to  send tp processor p
    *    - morton_to_receive: the morton indices to  exchange with processor p
    *
     * @tparam distribution_type 
     * @tparam iterator_type 
     * @tparam vector_vector_struct_type 
     * @tparam vector_vector_type 
     * @param[in] comm   the mpi communicator
     * @param[in] begin_left_ghost The iterator of the first ghost on the left
     * @param[in] end_left_ghost  The iterator of the last ghost on the left
     * @param[in] begin_right_ghost   The iterator of the first ghost on the right
     * @param[in] end_right_ghost The iterator of the last ghost on the right
     * @param[in] distrib  the data distribution 
     * @param[out] nb_messages_to_receive the number of morton indices to exchange with processor p
     * @param[out] nb_messages_to_send the number of morton indices to send tp processor p
     * @param[out] leaf_to_receive_access For each component a direct access to it (iterator on group, position into the group)
     * @param[out] morton_to_receive for each process the vector of Morton indexes to receive
    */
    template<typename distribution_type, typename iterator_type, typename vector_vector_struct_type,
             typename vector_vector_type>
    inline void start_step1(comm_type& comm, iterator_type begin_left_ghost, iterator_type end_left_ghost,
                            iterator_type begin_right_ghost, iterator_type end_right_ghost,
                            distribution_type const& distrib, std::vector<int>& nb_messages_to_receive,
                            std::vector<int>& nb_messages_to_send, vector_vector_struct_type& leaf_to_receive_access,
                            vector_vector_type& morton_to_receive, bool verbose = false)
    {
        // We iterate on the ghosts

        // function to fill the struture to_receive the cells for groups between first_group_ghost and last_group_ghost
        auto build_receive = [&nb_messages_to_receive, &leaf_to_receive_access, &distrib, &morton_to_receive,
                              verbose](auto first_group_ghost, auto last_group_ghost)
        {
            // if(verbose)
            //     std::cout << "step1 build_receive " << std::distance(first_group_ghost, last_group_ghost) << std::endl
            //               << std::flush;
            for(auto grp_ptr = first_group_ghost; grp_ptr != last_group_ghost; ++grp_ptr)
            {
                int idx{0};
                // if(verbose)
                //     std::cout << "      idx=   " << idx << std::endl << std::flush;
                // iterate on the cells
                for(auto const& component: (*grp_ptr)->components())
                {
                    auto morton = component.csymbolics().morton_index;
                    auto i = parallel::utils::get_proc_id(morton, distrib);
                    ++nb_messages_to_receive[i];
                    leaf_to_receive_access[i].push_back(std::make_pair(grp_ptr, idx));
                    morton_to_receive[i].push_back(morton);
                    // if(verbose)
                    //     std::cout << "     step 1    " << idx << "  " << *grp_ptr << "  " << morton << "  proc " << i
                    // << std::endl;
                    ++idx;
                }
            }
        };
        // Start on the left ghosts
        // if(verbose)
        //     std::cout << "    Left \n" << std::flush;
        if(std::distance(begin_left_ghost, end_left_ghost) > 0)
        {
            build_receive(begin_left_ghost, end_left_ghost);
        }
        // Start on the ghosts on the right
        // if(verbose)
        //     std::cout << "    right \n" << std::flush;

        if(std::distance(begin_right_ghost, end_right_ghost) > 0)
        {
            build_receive(begin_right_ghost, end_right_ghost);
        }
        // else
        // {
        //     std::cout << "      No ghost group" << std::endl << std::flush;
        // }
        // Faut-il les trier ???
        int p{0};
        // if(verbose)
        //     io::print("step 1  nb_messages_to_receive ", nb_messages_to_receive);
        // if(verbose)
        //     std::cout << " step1   morton_to_receive.size() " << morton_to_receive.size() << std::endl;
        for(auto& vec: morton_to_receive)
        {
            //   auto last = std::unique(vec.begin(), vec.end());
            //   vec.erase(last, vec.end());
            if(verbose)
                io::print("step 1    morton_to_receive[" + std::to_string(p++) + "] ", vec);
        }

        p = 0;
        // if(verbose)
        //     io::print("step 1  nb_messages_to_send ", nb_messages_to_send);
        // if(verbose)
        // {
        //     for(auto& vec: morton_to_receive)
        //     {
        //         io::print("step 1    morton_to_receive[" + std::to_string(p++) + "] ", vec);
        //     }
        // }

        ////////////////////
        /// Exchange the morton indexes with processor p
        auto mpi_int_type = cpp_tools::parallel_manager::mpi::get_datatype<int>();
        if(verbose)
            io::print("step 1  nb_messages_to_receive ", nb_messages_to_receive);
        // comm.barrier();
        // std::clog << "comm.alltoall\n";
        comm.alltoall(nb_messages_to_receive.data(), 1, mpi_int_type, nb_messages_to_send.data(), 1, mpi_int_type);
        // std::clog << "znd comm.alltoall\n";

        if(verbose)
            io::print("end step 1    nb_messages_to_send ", nb_messages_to_send);
    }

    /**
     * @brief We can now exchange the morton indices (STEP 2)
     * 
     *  Morton's list of indices to send their data (mutipoles/particles) to proc p
     * @tparam vector_vector_type 
     * @param[in] nb_proc number of mpi processes
     * @param[in] rank the mpi rank
     * @param[in] comm the communicator
     * @param[in] nb_messages_to_receive for each process the number of message to receive
     * @param[in] nb_messages_to_send  for each process the number of message to send
     * @param[in] morton_to_receive for  each process the vector of Morton indexes to receive
     * @param[out] morton_to_send for each process the vector of Morton indexes to send
    */
    template<typename vector_vector_type>
    inline void start_step2(int const& nb_proc, int const& rank, comm_type& comm,
                            std::vector<int>& nb_messages_to_receive, std::vector<int>& nb_messages_to_send,
                            vector_vector_type& morton_to_receive, vector_vector_type& morton_to_send,
                            bool verbose = false)
    {
        using mortonIdx_type = typename vector_vector_type::value_type::value_type;
        std::vector<cpp_tools::parallel_manager::mpi::request> tab_mpi_status;
        //
        auto mpi_morton_type = cpp_tools::parallel_manager::mpi::get_datatype<mortonIdx_type>();
        // io::print(" step2    nb_messages_to_send ", nb_messages_to_send);
        for(auto p = 0; p < nb_proc; ++p)
        {
            // if(p == rank)
            // {
            //     continue;
            // // }
            // std::cout << "proc p " << p << "nb_messages_to_send " << nb_messages_to_send[p] << std::endl;
            // std::cout << "proc p " << p << "nb_messages_to_send " << nb_messages_to_receive[p] << std::endl;

            // send the morton indexes morton_to_receive
            if(nb_messages_to_receive[p] != 0)
            {
                // io::print(" send to " + std::to_string(p) + " morton  ", morton_to_receive[p]);

                // io::print("step 2    morton_to_receive[" + std::to_string(p) + "] ", morton_to_receive[p]);

                comm.isend(morton_to_receive[p].data(), nb_messages_to_receive[p], mpi_morton_type, p, 600);
            }
        }
        // std::cout << std::endl << std::endl;
        // std::cout << " step2   Start receive communications \n";

        for(auto p = 0; p < nb_proc; ++p)
        {
            // std::cout << "proc p " << p << " nb_messages_to_receive " << nb_messages_to_receive[p] << std::endl;
            // std::cout << "proc p " << p << " nb_messages_to_send " << nb_messages_to_send[p] << std::endl;
            morton_to_send[p].resize(nb_messages_to_send[p], -1);
            // io::print("start_step2 init morton_to_send ", morton_to_send[p]);
            // Get the morton index to send
            if(nb_messages_to_send[p] != 0)
            {
                // std::cout << "step 2 me " << rank << " receive to " << p
                //           << " nb_messages_to_receive= " << nb_messages_to_receive[p]
                //           << " nb_messages_to_send= " << nb_messages_to_send[p] << std::endl
                //           << std::flush;

                tab_mpi_status.push_back(
                  comm.irecv(morton_to_send[p].data(), nb_messages_to_send[p], mpi_morton_type, p, 600));
            }
        }
        if(tab_mpi_status.size() > 0)
        {
            cpp_tools::parallel_manager::mpi::request::waitall(tab_mpi_status.size(), tab_mpi_status.data());
        }
        // // check
        // for(auto p = 0; p < nb_proc; ++p)
        // {
        //     io::print("step 2    morton_to_send[" + std::to_string(p) + "] ", morton_to_send[p]);
        // }
    }
    /**
     * @brief For the vector of Morton indices to be sent to processor p, we construct a direct access to the component (cell or leaf)
     * 
     * @tparam iterator_type 
     * @tparam vector_vector_struct_type 
     * @tparam vector_vector_type 
     * @param nb_proc the number of processors
     * @param begin_grp the first iterator on the group
     * @param end_grp the last iterator on the group
     * @param component_access the access to the component (iterator on group, position into the group)
     * @param morton_to_send for each processor the vector of Morton indexes to send
     */
    template<typename iterator_type, typename vector_vector_struct_type, typename vector_vector_type>
    auto build_direct_access_to_components(const int nb_proc, iterator_type begin_grp, iterator_type end_grp,
                                           vector_vector_struct_type& component_access,
                                           vector_vector_type const& morton_to_send) -> void
    {
        using access_type = typename vector_vector_struct_type::value_type;
        using vector_morton_type = typename vector_vector_type::value_type;

        bool verbose{false};
        auto build_index_grp =
          [&verbose](auto begin_grp, auto end_grp, vector_morton_type const& morton_to_send_p, access_type& to_send_p)
        {
            int idx{0};
            int max_idx = morton_to_send_p.size();
            to_send_p.resize(max_idx);
            // loop on the groups
            // auto it = std::begin(buffer);

            for(auto grp_ptr = begin_grp; grp_ptr != end_grp; ++grp_ptr)
            {
                int start_grp{0};
                auto const& csymb = (*grp_ptr)->csymbolics();
                // iterate on the cells
                // if(verbose)
                //   std::clog << idx << " morton "<< morton_to_send_p[idx] << " in [ "<<csymb.starting_index<< ", " << csymb.ending_index << "[\n";
                while(idx < max_idx and math::between(morton_to_send_p[idx], csymb.starting_index, csymb.ending_index))
                {   // find cell inside the group
                    int pos{-1};
                    for(int i = start_grp; i < (*grp_ptr)->size(); ++i)
                    {
                        auto morton = (*grp_ptr)->component(i).csymbolics().morton_index;
                        if(morton_to_send_p[idx] == morton)
                        {
                            pos = i;
                            start_grp = i + 1;
                            to_send_p[idx].first = grp_ptr;
                            to_send_p[idx].second = i;
                            // if (verbose)
                            //   std::clog << "  m= "<<morton << "  ptr " <<  to_send_p[idx].first->get() << " pos " << i << std::endl;
                            break;
                        }
                    }
                    ++idx;
                }
            }
            if(idx != max_idx)
            {
                std::cerr << "Didn't found the good number of morton indexes\n";
                // std::string outName2("bug_direct_access_to_component_rank_" + std::to_string(rank) + ".txt");
                // std::ofstream out(outName2);
                // scalfmm::io::trace(out, letGroupTree, 2);
                // out << "\n" << " \n";
                io::print(std::cerr, "morton_to_send: ", morton_to_send_p);
                std::cerr << "\n missing morton: ";
                // out << "missing morton: ";

                for(int i = idx; i < max_idx; ++i)
                {
                    std::cerr << morton_to_send_p[i] << " ";
                    // out << morton_to_send_p[i] << " ";
                }
                std::cerr << "\n";
                // out.close();
                throw std::runtime_error(" Missing morton index in building direct compononent access");
            }
        };

        for(auto p = 0; p < nb_proc; ++p)
        {
            // io::print(std::clog, "    morton_to_send[" + std::to_string(p) + "] ", morton_to_send[p]);
            if(morton_to_send[p].size() != 0)
            {
                // verbose = p == 3 ? true : false;
                build_index_grp(begin_grp, end_grp, morton_to_send[p], component_access[p]);
                // if(p == 3)
                // {
                //     auto const& elt = component_access[p];
                //     for(auto i = 0; i < elt.size(); ++i)
                //     {
                //         std::clog
                //           << "   -->p=" << p << " ptr="
                //           << elt[i].first->get()
                //           //	  << " m=" << (*(elt[i].first))->component(elt[i].second).csymbolics().morton_index
                //           << "  pos=" << elt[i].second << " m="
                //           << morton_to_send[p][i]
                //           //		  << " nb part "   << (*(elt[i].first))->component(elt[i].second).size()
                //           << std::endl;
                //     }
                // }
            }
        }
    }

    /**
     * @brief Construct the MPI type of the particle according to leaf_to_access
     * 
     * @tparam dimension 
     * @tparam vector_vector_struct_type 
     * @param leaf_to_access  For each processor the leaf to access (for receiving or sending)
     * @param mpi_position_type  the MPI type of the coordinate of the points of the particles
     * @param mpi_input_type   the MPI type of the inputs of the particles
     * @return std::vector<MPI_Datatype> 
     */
    template<std::size_t dimension, typename vector_vector_struct_type>
    auto inline build_mpi_particles_type(vector_vector_struct_type const& leaf_to_access, int const nb_inputs,
                                         MPI_Datatype mpi_position_type,
                                         MPI_Datatype mpi_input_type) -> std::vector<MPI_Datatype>
    {
        const int nb_proc{int(leaf_to_access.size())};
        std::vector<MPI_Datatype> newtype(nb_proc);

        for(auto p = 0; p < nb_proc; ++p)
        {
            if(leaf_to_access[p].size() != 0)
            {
                // leaf_to_access[p] = std::vector<pair> [i] = (group_ptr, index_in_group)
                auto const& elt = leaf_to_access[p];
                int nb_mpi_types{int(elt.size() * (dimension + nb_inputs))};
                std::vector<int> length(nb_mpi_types, 1);
                std::vector<MPI_Aint> disp(nb_mpi_types);
                std::vector<MPI_Datatype> type(nb_mpi_types);
                int nb_elt{int(elt.size())};

                int size_msg{0};
                for(auto i = 0; i < elt.size(); ++i)
                {
                    int jump{0};
                    auto leaf = (*(elt[i].first))->component(elt[i].second);
                    // tuple of iterators
                    // leaf[0] return a particle proxy on the first particle
                    auto proxy_position = leaf[0].position();
                    auto ptr_x = &(proxy_position[0]);

                    int nb_elt_leaf{int(leaf.size())};
                    size_msg += nb_elt_leaf;
                    MPI_Type_contiguous(nb_elt_leaf, mpi_position_type, &type[i]);
                    MPI_Get_address(&(proxy_position[0]), &disp[i]);
                    jump += nb_elt;
                    for(int k = 1; k < dimension; ++k, jump += nb_elt)
                    {
                        type[i + jump] = type[i];
                        MPI_Get_address(&(proxy_position[k]), &disp[i + jump]);
                    }
                    // get the inputs
                    auto ptr_inputs_0 = &(leaf[0].inputs(0));
                    MPI_Type_contiguous(nb_elt_leaf, mpi_input_type, &type[i + jump]);
                    MPI_Get_address(ptr_inputs_0, &disp[i + jump]);
                    jump += nb_elt;
                    for(int k = 1, stride = dimension + 1; k < nb_inputs; ++k, ++stride)
                    {
                        type[i + stride * nb_elt] = type[i + dimension * nb_elt];
                        MPI_Get_address(&(leaf[0].inputs(k)), &disp[i + stride * nb_elt]);
                    }
                    // std::cout << p << " " << leaf.csymbolics().morton_index << " nb part " << leaf.size() << " *ptr_x "
                    //           << proxy_position << " snd part " << *(ptr_x + 1) << " inputs0: " << leaf[0].inputs()[0]
                    //           << " inputs1: " << *(&(leaf[0].inputs()[0]) + 1) << "  ptr " << *(ptr_inputs_0 + 1)
                    //           << std::endl;
                }   // end loop on leaf_view
                // std::cout << " create type " << std::endl;
                // io::print("  " + std::to_string(p) + " disp", disp);
                MPI_Type_create_struct(nb_mpi_types, length.data(), disp.data(), type.data(), &newtype[p]);
                MPI_Type_commit(&newtype[p]);
                // std::cout << "  send to " << p << " size " << size_msg << std::endl;
            }
        }
        return newtype;
    }
    /// @brief Construct the MPI type of all multipoles to send to a different process
    /// @tparam vector_vector_struct_type
    /// @param cell_to_access
    /// @param nb_inputs
    /// @param mpi_multipole_type
    /// @return A vector of MPI type
    template<typename vector_vector_struct_type>
    auto inline build_mpi_multipoles_type(vector_vector_struct_type const& cell_to_access, int const nb_inputs,
                                          MPI_Datatype& mpi_multipole_type) -> std::vector<MPI_Datatype>
    {
        // std::cout << " build_mpi_multipoles_type inside nb_inputs" << nb_inputs << std::endl << std::flush;

        const int nb_proc{int(cell_to_access.size())};
        std::vector<MPI_Datatype> newtype(nb_proc, MPI_DATATYPE_NULL);

        for(auto p = 0; p < nb_proc; ++p)
        {
            // std::clog << "   multipole type(p=" << p << ") nb cells to pack " << cell_to_access[p].size() << "\n";
            // std::clog << std::flush;
            if(cell_to_access[p].size() != 0)
            {
                auto const& elt = cell_to_access[p];
                // number of mpi type to construct (=cells number * nb_inputs)
                int nb_mpi_types{int(elt.size() * nb_inputs)};
                //
                std::vector<int> length(nb_mpi_types, 1);
                std::vector<MPI_Aint> disp(nb_mpi_types);
                std::vector<MPI_Datatype> type(nb_mpi_types);
                // if(p == 3)
                // {
                //     //bug
                //     for(auto i = 0; i < elt.size(); ++i)
                //     {
                //         std::clog << " ptr: " << *(elt[i].first) << " pos " << elt[i].second << std::endl;
                //     }
                // std::clog << "------\n";
                // }
                int size_msg{0};
                for(auto i = 0; i < elt.size(); ++i)
                {
                    // *(elt[i].first) = ptr_group
                    // elt[i].second = index inside group
                    int jump{0};
                    auto const& cell = (*(elt[i].first))->component(elt[i].second);
                    // tuple of iterators
                    // cell[0] return a particle proxy on the first particle
                    auto const& m = cell.transfer_multipoles();
                    auto nb_m = m.size();   // get number of multipoles = nb_inputs
                    // std::cout << "          nb_m" << m.size() << std::endl << std::flush;
                    for(std::size_t k{0}; k < nb_m; ++k)
                    {
                        auto const& ten = m.at(k);
                        // std::cout << "     size " << int(ten.size()) << std::endl << std::flush;
                        MPI_Type_contiguous(int(ten.size()), mpi_multipole_type, &type[i * nb_inputs + k]);
                        //                    MPI_Get_address(&(ten.data()[0]), &disp[i * nb_inputs + k]);
                        MPI_Get_address(ten.data(), &disp[i * nb_inputs + k]);
                        // std::cout << "     i * nb_inputs + k " << i * nb_inputs + k << " nb_mpi_types " << nb_mpi_types
                        //           << std::endl
                        //           << std::flush;
                    }
                }   // end loop on cell_view
                //	    io::print(std::clog, "m2l(type) disp: ", disp); std::clog << std::flush;
                //	    io::print(std::clog, "m2l(type) length: ", length); std::clog << std::flush;

                MPI_Type_create_struct(nb_mpi_types, length.data(), disp.data(), type.data(), &newtype[p]);
                MPI_Type_commit(&newtype[p]);
                //	    std::clog << std::flush;
            }
        }
        return newtype;
    }
    template<typename vector_vector_struct_type>
    auto inline build_mpi_multipoles_type2(vector_vector_struct_type const& cell_to_access, int const nb_inputs,
                                           MPI_Datatype mpi_multipole_type) -> std::vector<MPI_Datatype>
    {
        const int nb_proc{int(cell_to_access.size())};
        std::vector<MPI_Datatype> newtype(nb_proc, MPI_DATATYPE_NULL);

        for(auto p = 0; p < nb_proc; ++p)
        {
            if(cell_to_access[p].size() != 0)
            {
                auto const& elt = cell_to_access[p];
                // number of mpi type to construct (=cells number * nb_inputs)
                int nb_mpi_types{int(elt.size() * nb_inputs)};
                //
                std::vector<int> length(nb_mpi_types);
                std::vector<MPI_Aint> disp(nb_mpi_types);
                std::vector<MPI_Datatype> type(nb_mpi_types, mpi_multipole_type);

                int size_msg{0};
                for(auto i = 0; i < elt.size(); ++i)
                {
                    // *(elt[i].first) = ptr_group
                    // elt[i].second = index inside group
                    int jump{0};
                    auto const& cell = (*(elt[i].first))->component(elt[i].second);
                    //
                    auto const& m = cell.transfer_multipoles();
                    auto nb_m = m.size();   // get number of mutilpoles = nb_inputs
                    //			    std::cout << "          nb_m" <<  m.size() <<std::endl;
                    for(std::size_t k{0}; k < nb_m; ++k)
                    {
                        auto const& ten = m.at(k);
                        // std::cout << " size " << int(ten.size()) << std::endl;
                        // MPI_Type_contiguous(int(ten.size()), mpi_multipole_type, &type[i * nb_inputs + k]);
                        length[i * nb_inputs + k] = ten.size();
                        MPI_Get_address(&(ten.data()[0]), &disp[i]);
                    }
                    // std::cout << p << " " << cell.csymbolics().morton_index << " nb part " << cell.size() << " *ptr_x "
                    //           << proxy_position << " snd part " << *(ptr_x + 1) << " inputs0: " << cell[0].inputs()[0]
                    //           << " inputs1: " << *(&(cell[0].inputs()[0]) + 1) << "  ptr " << *(ptr_inputs_0 + 1)
                    //           << std::endl;
                }   // end loop on cell_view
                // std::cout << " create type " << std::endl;
                // io::print("  " + std::to_string(p) + " disp", disp);
                MPI_Type_create_struct(nb_mpi_types, length.data(), disp.data(), type.data(), &newtype[p]);
                MPI_Type_commit(&newtype[p]);
                // std::cout << "  send to " << p << " size " << size_msg << std::endl;
            }
        }
        return newtype;
    }

    template<typename Tree>
    void prepare_comm_up(Tree tree) {

    };

}   // namespace scalfmm::parallel::comm
