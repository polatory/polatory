// --------------------------------
// See LICENCE file at project root
// File : algorithm/omp/upward.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_MPI_TRANSFER_HPP
#define SCALFMM_ALGORITHMS_MPI_TRANSFER_HPP

#ifdef _OPENMP

#include <iostream>
#include <map>
#include <omp.h>
#include <ostream>
#include <utility>

#include "scalfmm/algorithms/omp/transfer.hpp"
#include "scalfmm/meta/traits.hpp"
#include "scalfmm/operators/tags.hpp"
#include "scalfmm/parallel/mpi/comm.hpp"
#include "scalfmm/parallel/mpi/utils.hpp"
#include "scalfmm/parallel/utils.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/massert.hpp"
#include "scalfmm/utils/math.hpp"

#include <cpp_tools/parallel_manager/parallel_manager.hpp>

namespace scalfmm::algorithms::mpi::pass
{
    template<typename ContainerType>
    void print_access(std::ostream& out, ContainerType& access)
    {
        // for(auto p = 0; p < cells_to_access.size(); ++p)
        // {
        //     auto& access = cells_to_access[p];

        out << "task_send_multipole_at_level for process   size " << access.size() << std::endl << std::flush;
        for(auto i = 0; i < access.size(); ++i)
        {
            out << "   task_send  " << i << "  ptr  " << std::flush << access[i].first->get() << "  morton "
                << (*(access[i].first))->component(access[i].second).csymbolics().morton_index << std::flush
                << "multipole " << (*(access[i].first))->component(access[i].second).transfer_multipoles().at(0)
                << std::endl
                << std::flush;
        }
        out << "---------------------------" << std::endl << std::flush;
        // }
    }
    /**
     * @brief Build the buffer of multipole (no more used)
     * 
     * @tparam IteratorType 
     * @tparam StructMorton 
     * @tparam BufferType 
     */
    template<typename IteratorType, typename VectorOfVectorMortonType, typename BufferType>
    auto build_buffer_func(IteratorType first_group_ghost, IteratorType last_group_ghost,
                           VectorOfVectorMortonType const& index_to_send, BufferType& buffer)
    {
        try
        {
            //		    std::cout <<  " -----      build_buffer   ----\n" <<std::flush;
            //		    std::cout <<  " -----      buffer size " <<buffer.size() <<std::endl<<std::flush;
            //		    std::cout <<  " -----     index_to_send size " <<index_to_send.size() <<std::endl<<std::flush;
            int idx{0};
            int max_idx = index_to_send.size();   // loop on the groups
            auto it = std::begin(buffer);
            for(auto grp_ptr = first_group_ghost; grp_ptr != last_group_ghost; ++grp_ptr)
            {
                int start_grp{0};

                auto const& csymb = (*grp_ptr)->csymbolics();
                // iterate on the cells
                while(idx < max_idx and math::between(index_to_send[idx], csymb.starting_index, csymb.ending_index))
                {   // find cell inside the group
                    int pos{-1};
                    for(int i = start_grp; i < (*grp_ptr)->size(); ++i)
                    {
                        auto morton = (*grp_ptr)->component(i).csymbolics().morton_index;
                        if(index_to_send[idx] == morton)
                        {
                            pos = i;
                            start_grp = i + 1;
                            // std::cout << "   pos = " << pos << std::endl;
                            break;
                        }
                    }
                    //			     std::cout << " morton to find " << index_to_send[idx] << " cell found "
                    //			               << (*grp_ptr)->component(pos).csymbolics().morton_index << '\n';
                    auto const& cell = (*grp_ptr)->component(pos);
                    auto const& m = cell.ctransfer_multipoles();
                    auto nb_m = m.size();
                    //			    std::cout << "          nb_m" <<  m.size() <<std::endl;
                    for(std::size_t i{0}; i < nb_m; ++i)
                    {
                        auto const& ten = m.at(i);
                        std::copy(std::begin(ten), std::end(ten), it);
                        it += ten.size();
                    }
                    ++idx;
                }
            }
            //		                  std::cout <<  " -----      build_buffer   ----\n" <<std::flush;
        }
        catch(std::exception& e)
        {
            std::cout << " error in buffer building !!!!!!!!!\n";
            std::cout << e.what() << '\n' << std::flush;
        }
    }
    template<typename BufferType, typename AccessType>
    auto build_buffer(const int nb_inputs, AccessType const& cells_to_send_access) -> BufferType
    {
        BufferType buffer;
        try
        {
            //number of cells x nb inputs x size of an input

            auto const& cell = (cells_to_send_access[0].first->get())->component(cells_to_send_access[0].second);
            const int multipoleSize{int(cell.transfer_multipoles().at(0).size())};

            const int buffer_size{int(cells_to_send_access.size()) * nb_inputs * multipoleSize};
            buffer.resize(buffer_size);
            std::cout << " -----      build_buffer   ----\n" << std::flush;
            std::cout << " -----      buffer size " << buffer.size() << std::endl << std::flush;
            auto it = std::begin(buffer);
            // iterate on the cells
            for(auto access: cells_to_send_access)
            {
                auto const& cell = (*(access.first))->component(access.second);

                //			     std::cout << " morton to find " << index_to_send[idx] << " cell found "
                //			               << (*grp_ptr)->component(pos).csymbolics().morton_index << '\n';
                auto const& m = cell.transfer_multipoles();
                auto nb_m = m.size();
                //			    std::cout << "          nb_m" <<  m.size() <<std::endl;
                for(std::size_t i{0}; i < nb_m; ++i)
                {
                    auto const& ten = m.at(i);
                    std::copy(std::begin(ten), std::end(ten), it);
                    it += ten.size();
                }
            }
            std::cout << " -----      build_buffer   ----\n" << std::flush;
            // io::print("buffer: ", buffer);
        }
        catch(std::exception& e)
        {
            std::cout << " error in buffer building !!!!!!!!!\n";
            std::cout << e.what() << '\n' << std::flush;
        }
        return buffer;
    }

    /// @brief  Perform the communications to send the multipoles

    /// @tparam valueType
    /// @tparam TreeS      the tree type
    /// @tparam VectorOfVectorMortonType
    /// @param level    the level in the tree
    /// @param tree     the tree containing the multipole to send
    /// @param morton_to_send  the morton index of the cells containing the multipole
    template<typename TreeS, typename VectorOfVectorMortonType>
    void task_send_multipole_at_level(int const& level, TreeS& tree, VectorOfVectorMortonType const& morton_to_send)
    {
        static constexpr auto prio{omp::priorities::max};
        static constexpr int nb_inputs = TreeS::cell_type::storage_type::inputs_size;
        using multipole_type = typename TreeS::base_type::cell_type::storage_type::transfer_multipole_type;

        using dependencies_type = typename TreeS::group_of_cell_type::symbolics_type::ptr_multi_dependency_type;
        std::vector<dependencies_type> dependencies;

        auto& para = tree.get_parallel_manager();
        auto comm = para.get_communicator();
        const auto nb_proc = para.get_num_processes();
        const auto rank = para.get_process_id();

        for(auto p = 0; p < nb_proc; ++p)
        {
            // std::cout << "  Morton to send to " << p << "   " << morton_to_send[p].size() << std::endl;
            // io::print(" morton_to_send[" + std::to_string(p) + "]", morton_to_send[p]);

            if(morton_to_send[p].size() > 0)
            {
                /// We first construct the in dependencies to ensure that multipoles
                /// are updated by the  previous pass.
                parallel::utils::build_multipoles_dependencies_from_morton_vector(
                  tree.begin_mine_cells(level), tree.end_mine_cells(level), morton_to_send[p], dependencies);
            }
        }
        std::sort(std::begin(dependencies), std::end(dependencies));
        auto last = std::unique(std::begin(dependencies), std::end(dependencies));
        dependencies.erase(last, dependencies.end());
        io::print("M2L-old dependencies(send): ", dependencies);
        // //
        // // build direct access to the cells whose multipoles are to be sent
        // //
        // using grp_access_type = std::pair<decltype(tree.begin_cells(level)), int>;
        // auto mpi_multipole_value_type = cpp_tools::parallel_manager::mpi::get_datatype<multipole_type>();
        // std::vector<std::vector<grp_access_type>> cells_to_send_access(nb_proc);
        // auto begin_grp = tree.begin_mine_cells(level);
        // auto end_grp = tree.end_mine_cells(level);
        // // std::clog << "          build_direct_access_to_components " << std::endl << std::flush;
        // scalfmm::parallel::comm::build_direct_access_to_components(nb_proc, begin_grp, end_grp, cells_to_send_access,
        //                                                            morton_to_send);

        //     for(auto p = 0; p < nb_proc; ++p)
        //   {
        //         auto& access = cells_to_send_access[p];

        //         std::cout << "task_send_multipole_at_level for process " << p << "  size " << access.size() << std::endl
        //                   << std::flush;
        //         for(auto i = 0; i < access.size(); ++i)
        //         {
        //             std::cout << "   task_send  " << i << "  ptr  " << std::flush << access[i].first->get() << "  index "
        //                       << access[i].second << "  morton " << std::flush
        //                       << (*(access[i].first))->component(access[i].second).csymbolics().morton_index
        //                       << " multipoles "
        //                       << (*(access[i].first))->component(access[i].second).transfer_multipoles().at(0) << std::endl
        //                       << std::flush;
        //         }
        //         std::cout << "---------------------------" << std::endl << std::flush;
        //     }
        static constexpr std::size_t inputs_size = TreeS::base_type::cell_type::inputs_size;
        //
        // construct the MPI type to send the all multipoles
        //
        // tree.get_send_multipole_types(level) = scalfmm::parallel::comm::build_mpi_multipoles_type(
        //   cells_to_send_access, inputs_size, mpi_multipole_value_type);

        // auto const& multipole_type_to_send = tree.get_send_multipole_types(level);
        // tree.print_send_multipole_types(level);
        // for(auto p = 0; p < nb_proc; ++p)
        // {
        //     std::cout << " m2l(prep) ptr_data_type(" << p << ") " << &(multipole_type_to_send[p]) << " level: " << level
        //               << std::endl
        //               << std::flush;
        // }
        //
        // std::clog << "          end build_mpi_multipoles_type " << std::endl << std::flush;
        auto ptr_tree = &tree;
//
// task to perform on communications ,  shared(morton_to_send)
#pragma omp task untied firstprivate(rank, nb_proc, level, dependencies, ptr_tree)                                     \
  depend(iterator(std::size_t it = 0 : dependencies.size()), in : (dependencies[it])[0]) priority(prio)
        {
            std::vector<cpp_tools::parallel_manager::mpi::request> send_mpi_status;

            std::cout << "m2l-old  task(send) " << std::endl << std::flush;
            io::print("m2l-old  task(send) dependencies(in) ", dependencies);
            // parallel::comm::print_all_cells(*ptr_tree, level, "M2L task(send)");
            auto morton_to_send1 = ptr_tree->send_morton_indexes(level);

            for(int p = 0; p < nb_proc; ++p)
            {
                io::print(" morton_to_send1-old  ", morton_to_send1[p]);
            }
            //
            // build direct access to the cells whose multipoles are to be sent
            //
            using grp_access_type = std::pair<decltype(tree.begin_cells(level)), int>;
            auto mpi_multipole_value_type = cpp_tools::parallel_manager::mpi::get_datatype<multipole_type>();
            std::vector<std::vector<grp_access_type>> cells_to_send_access(nb_proc);
            auto begin_grp = ptr_tree->begin_mine_cells(level);
            auto end_grp = ptr_tree->end_mine_cells(level);
            // std::clog << "          build_direct_access_to_components " << std::endl << std::flush;
            scalfmm::parallel::comm::build_direct_access_to_components(nb_proc, begin_grp, end_grp,
                                                                       cells_to_send_access, morton_to_send1);
            // tree.get_send_multipole_types(level) = scalfmm::parallel::comm::build_mpi_multipoles_type(
            //   cells_to_send_access, inputs_size, mpi_multipole_value_type);
            auto multipole_type_to_send = scalfmm::parallel::comm::build_mpi_multipoles_type(
              cells_to_send_access, inputs_size, mpi_multipole_value_type);
            // auto const& multipole_type_to_send = ptr_tree->get_send_multipole_types(level);
            tree.print_send_multipole_types(level);

            for(auto p = 0; p < nb_proc; ++p)
            {
                if(morton_to_send1[p].size() > 0)
                // if(multipole_type_to_send[p] != MPI_DATATYPE_NULL)
                {
                    // print_access(std::cout, cells_to_send_access[p]);

                    std::cout << "m2l-old  task(send) send to " << p << std::endl << std::flush;
                    std::cout << " m2l-old (task) ptr_data_type(" << p << ") " << &(multipole_type_to_send[p])
                              << " level: " << level << std::endl
                              << std::flush;

                    send_mpi_status.push_back(comm.isend(MPI_BOTTOM, 1, multipole_type_to_send[p], p, 611));

                    std::cout << " m2l(task)-old  end send to " << p << "\n" << std::flush;
                }
            }

            std::cout << " m2l(task)-old  end task \n" << std::flush;

        }   // end task
    }
    /// @brief  Receive the multipoles and put them in ghost groups
    ///
    /// @tparam TreeS
    /// @tparam VectorOfVectorGroupAccessType
    /// @param level  level ine tree
    /// @param tree   the tree
    /// @param cells_to_receive_access  Cell access vector (ptr on the gout, index inside it)
    template<typename TreeS, typename VectorOfVectorGroupAccessType>
    void task_receive_multipole_at_level(int const& level, TreeS& tree,
                                         VectorOfVectorGroupAccessType const& cells_to_receive_access)
    {
        // We first construct the out dependencies (all the ghost groups (right and left))
        //  naive version
        //
        auto& para = tree.get_parallel_manager();
        auto comm = para.get_communicator();
        const auto nb_proc = para.get_num_processes();
        //
        auto size_dep{std::distance(tree.begin_cells(level), tree.begin_mine_cells(level)) +
                      std::distance(tree.end_mine_cells(level), tree.end_cells(level))};
        using dependencies_type = typename TreeS::group_of_cell_type::symbolics_type::ptr_multi_dependency_type;
        std::vector<dependencies_type> dependencies(size_dep);
        int idx{0};
        for(auto it_grp = tree.begin_cells(level); it_grp != tree.begin_mine_cells(level); ++it_grp, ++idx)
        {
            dependencies[idx] = &(it_grp->get()->ccomponent(0).cmultipoles(0));
        }
        for(auto it_grp = tree.end_mine_cells(level); it_grp != tree.end_cells(level); ++it_grp, ++idx)
        {
            dependencies[idx] = &(it_grp->get()->ccomponent(0).cmultipoles(0));
        }
        io::print("dependencies(recv)-old : ", dependencies);
        //
        // construct the MPI type to send the all multipoles
        //
        using multipole_type = typename TreeS::base_type::cell_type::storage_type::transfer_multipole_type;
        auto mpi_multipole_value_type = cpp_tools::parallel_manager::mpi::get_datatype<multipole_type>();
        static constexpr std::size_t inputs_size = TreeS::base_type::cell_type::inputs_size;
        // auto multipole_type_to_receive = scalfmm::parallel::comm::build_mpi_multipoles_type(
        //   cells_to_receive_access, inputs_size, mpi_multipole_value_type);
        tree.get_receive_multipole_types(level) = scalfmm::parallel::comm::build_mpi_multipoles_type(
          cells_to_receive_access, inputs_size, mpi_multipole_value_type);

        // auto ptr_multipole_type_to_receive = tree.get_multipole_types(level).data();
        //receive the multipoles
        // tree.set_receive_access(to_receive);
        static constexpr auto prio{omp::priorities::max};
        auto ptr_tree = &tree;

#pragma omp task firstprivate(nb_proc, ptr_tree)                                                                       \
  depend(iterator(std::size_t it = 0 : dependencies.size()), inout : (dependencies[it])[0]) priority(prio)
        {
            std::cout << "M2L-old  task(transfer(receiv)) " << std::endl << std::flush;
            io::print("M2L-old  transfer_comm(task) dependencies(in): ", dependencies);

            std::vector<cpp_tools::parallel_manager::mpi::request> recept_mpi_status;
            auto ptr_multipole_type_to_receive = ptr_tree->get_receive_multipole_types(level).data();
            for(auto p = 0; p < nb_proc; ++p)
            {
                if(ptr_multipole_type_to_receive[p] != MPI_DATATYPE_NULL)
                {
                    recept_mpi_status.push_back(comm.irecv(MPI_BOTTOM, 1, ptr_multipole_type_to_receive[p], p, 611));
                }
            }
            if(recept_mpi_status.size() > 0)
            {
                cpp_tools::parallel_manager::mpi::request::waitall(recept_mpi_status.size(), recept_mpi_status.data());
                {
                    std::cout << "M2L-old   --  level " << level << "   --  " << std::endl;
                    scalfmm::component::for_each_mine_component(tree.begin_cells(level), tree.end_cells(level),
                                                                [](auto const& cell)
                                                                {
                                                                    std::cout << "M2L task(end receive) cell index "
                                                                              << cell.index() << "  multipoles "
                                                                              << cell.transfer_multipoles().at(0)
                                                                              << "  locals " << cell.locals().at(0)
                                                                              << std::endl
                                                                              << std::flush;
                                                                });
                }
            }
            std::clog << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n" << std::flush;

        }   // end task
    }

    /// @brief generate a task to send/receive the multipoles
    ///
    /// The received multipoles are put in ghost groups.
    /// @tparam TreeS
    /// @param level the level in the tree
    /// @param tree  /the tree containing the source particles
    template<typename TreeS, typename TreeT>
    void inline task_communications(const int level, TreeS& tree, TreeT& treeT)
    {
        // std::cout << "task_communications  fct start\n ";
        static constexpr auto prio{omp::priorities::max - 5};
        static constexpr int nb_inputs = TreeS::cell_type::storage_type::inputs_size;
        //
        using multipole_type = typename TreeS::base_type::cell_type::storage_type::transfer_multipole_type;
        using dependencies_type = typename TreeS::group_of_cell_type::symbolics_type::ptr_multi_dependency_type;
        using grp_access_type = std::pair<decltype(tree.begin_cells(level)), int>;

        //
        auto& para = tree.get_parallel_manager();
        const auto nb_proc = para.get_num_processes();
        const auto rank = para.get_process_id();
        if(nb_proc == 1)
        {
            return;
        }
        //
        auto size_dep{std::distance(tree.begin_cells(level), tree.begin_mine_cells(level)) +
                      std::distance(tree.end_mine_cells(level), tree.end_cells(level))};
        std::vector<dependencies_type> dependencies_in, dependencies_out(size_dep);

        // Build the dependencies in and out
        {
            /// We first construct the in dependencies to ensure that multipoles
            /// are updated by the  previous pass.
            auto const& morton_to_send = tree.send_morton_indexes(level);
            // io::print("m2l(task comm) morton_to_sendl=" + std::to_string(level) + "): ", morton_to_send);

            for(auto p = 0; p < nb_proc; ++p)
            {
                if(morton_to_send[p].size() > 0)
                {
                    parallel::utils::build_multipoles_dependencies_from_morton_vector(
                      tree.begin_mine_cells(level), tree.end_mine_cells(level), morton_to_send[p], dependencies_in);
                }
            }
            std::sort(std::begin(dependencies_in), std::end(dependencies_in));
            auto last = std::unique(std::begin(dependencies_in), std::end(dependencies_in));
            dependencies_in.erase(last, dependencies_in.end());
            //
            // out dependencies (on all groups of ghosts)
            //
            int idx{0};
            for(auto it_grp = tree.begin_cells(level); it_grp != tree.begin_mine_cells(level); ++it_grp, ++idx)
            {
                dependencies_out[idx] = &(it_grp->get()->ccomponent(0).cmultipoles(0));
            }
            for(auto it_grp = tree.end_mine_cells(level); it_grp != tree.end_cells(level); ++it_grp, ++idx)
            {
                dependencies_out[idx] = &(it_grp->get()->ccomponent(0).cmultipoles(0));
            }
            // io::print("m2l(task comm) dependencies(transfer)(out): ", dependencies_out);
        }
        // std::cout << " insert task (comm M2L(level=" + std::to_string(level) + ") \n";
        // io::print("      out:   ", dependencies_out);
        // io::print("      in:    ", dependencies_in);

        auto ptr_tree = &tree;
        auto ptr_treeT = &treeT;
        // std::cout << "      inout: " << ptr_tree << std::endl;   //to serialise the communication tasks

        // clang-format off
	#pragma omp task untied default(none) firstprivate(ptr_tree, nb_proc, rank, nb_inputs, level)  shared(std::cerr, std::clog) \
        depend(iterator(std::size_t it = 0 : dependencies_in.size()), in : (dependencies_in[it])[0])      \
        depend(iterator(std::size_t it = 0 : dependencies_out.size()), out : (dependencies_out[it])[0]) \
        depend(inout: ptr_tree[0], ptr_treeT[0] ) priority(prio)
        // clang-format on
        {
            int receive_section{0}, send_section{0};
            // std::clog << " m2l task(comm(l=" << level << ")) Send part \n" << std::flush;
            // #ifndef NO_COMM_TASK
            try
            {
                const int tag_level = 611 + level;
                send_section = 1;
                auto mpi_multipole_value_type = cpp_tools::parallel_manager::mpi::get_datatype<multipole_type>();
                auto comm = ptr_tree->get_parallel_manager().get_communicator();
                // send part

                auto& morton_to_send = ptr_tree->send_morton_indexes(level);
                // construct cells access (send)
                //                std::vector<std::vector<grp_access_type>> cells_to_send_access(nb_proc);
                auto& cells_to_send_access = ptr_tree->send_cells_access(level);
                auto begin_grp = ptr_tree->begin_mine_cells(level);
                auto end_grp = ptr_tree->end_mine_cells(level);
                // std::clog << " m2l task(comm(l=" << level << ")) build direct acces \n" << std::flush;
                scalfmm::parallel::comm::build_direct_access_to_components(nb_proc, begin_grp, end_grp,
                                                                           cells_to_send_access, morton_to_send);
                // build type to send all the multipoles
                // std::clog << " m2l task(comm(l=" << level << ")) build type \n" << std::flush;
                auto multipole_type_to_send = scalfmm::parallel::comm::build_mpi_multipoles_type(
                  cells_to_send_access, nb_inputs, mpi_multipole_value_type);
                //
                for(auto p = 0; p < nb_proc; ++p)
                {
                    // io::print("morton_to_send", morton_to_send[p]);
                    if(morton_to_send[p].size() > 0)
                    {
                        // std::clog << "m2l task(send(l=" << level << ")) send to " << p << " tag " << tag_level
                        //           << std::endl
                        //           << std::flush;
                        // std::cout << "m2l task(send) ptr_data_type(" << p << ") "   //<< &(multipole_type_to_send[p])
                        //           << " level: " << level << std::endl
                        //           << std::flush;
                        // io::print(std::cout, "morton_to_send(" + std::to_string(p) + ")", std::begin(morton_to_send[p]),
                        //           std::end(morton_to_send[p]));
                        // std::cout << std::flush << std::endl;
                        // #ifdef COMM_SEND
                        comm.isend(MPI_BOTTOM, 1, multipole_type_to_send[p], p, tag_level);
                        // #endif
                    }
                }
                // std::clog << "  m2l task(comm(l=" << level << ")) send end \n";
                ///  end send part
                ///////////////////////////////////////////////////////
                receive_section = 1;
                {
                    // std::clog << " m2l task(comm(l=" << level << ")) Receive part  level " << level << std::endl;
                    // receive part
                    auto& cells_to_receive_access = ptr_tree->receive_cells_access(level);
                    // for(auto p = 0; p < nb_proc; ++p)
                    // {
                    //     auto& access = cells_to_receive_access.at(p);
                    //     if(access.size())
                    //     {
                    //         std::cout << "  cells_to_receive_access " << p << "  size " << access.size() << std::endl
                    //                   << std::flush;
                    //         for(auto i = 0; i < access.size(); ++i)
                    //         {
                    //             std::cout << i << "  ptr  " << access[i].first->get() << "  index " << access[i].second
                    //                       << "  morton "
                    //                       << (*(access[i].first))->component(access[i].second).csymbolics().morton_index
                    //                       << std::endl;
                    //         }
                    //     }
                    // }
                    //
                    auto type = scalfmm::parallel::comm::build_mpi_multipoles_type(cells_to_receive_access, nb_inputs,
                                                                                   mpi_multipole_value_type);

                    auto ptr_multipole_type_to_receive = ptr_tree->get_receive_multipole_types(level);

                    // post receive
                    std::vector<cpp_tools::parallel_manager::mpi::request> recept_mpi_status;
                    for(auto p = 0; p < nb_proc; ++p)
                    {
                        if(cells_to_receive_access.at(p).size() != 0)
                        // if(ptr_multipole_type_to_receive[p] != MPI_DATATYPE_NULL)
                        {
                            // std::clog << "m2l task(comm(l=" << level << ")) post ireceive from " << p << " tag "
                            //           << tag_level << std::endl
                            //           << std::flush;
                            recept_mpi_status.push_back(comm.irecv(MPI_BOTTOM, 1, type[p], p, tag_level));
                        }
                    }

                    if(recept_mpi_status.size() > 0)
                    {
#ifndef SCALFMM_USE_WAIT_ANY
                        int cpt{0};
                        while(cpt != recept_mpi_status.size())
                        {
                            int count{1};
                            int index{-1};
                            MPI_Status status;
                            MPI_Waitany(int(recept_mpi_status.size()),
                                        reinterpret_cast<MPI_Request*>(recept_mpi_status.data()), &index, &status);
                            // index =  cpp_tools::parallel_manager::mpi::request::waitany(recept_mpi_status.size()), recept_mpi_status.data(),status);
                            ++cpt;
                            // std::clog << "receive one comm from " << status.MPI_SOURCE << " tag " << status.MPI_TAG
                            //           << " wait for " << recept_mpi_status.size() - cpt << std::endl;

                            if(status.MPI_TAG != tag_level)
                            {
                                std::cerr << "   wrong tag  wait for " << tag_level << " received " << status.MPI_TAG
                                          << std::endl;
                                --cpt;
                            }
                        }
#else
                        // std::clog << " m2l task(comm(l=" << level << ")) wait all level " << level << "  \n"
                        //           << std::flush;
                        cpp_tools::parallel_manager::mpi::request::waitall(recept_mpi_status.size(),
                                                                           recept_mpi_status.data());

#endif
                        // std::cout << " m2l task(comm) wait all level " << level << "  \n" << std::flush;
                        // std::clog << " m2l task(comm) wait all level " << level << "  \n" << std::flush;

                        // {
                        //     std::cout << "M2L  --  level " << level << "   --  " << std::endl;
                        //     scalfmm::component::for_each_mine_component(
                        //       ptr_tree->begin_cells(level), ptr_tree->end_cells(level),
                        //       [](auto const& cell)
                        //       {
                        //           std::cout << "M2L task(end receive) cell index " << cell.index() << "  multipoles "
                        //                     << cell.transfer_multipoles().at(0) << "  locals " << cell.locals().at(0)
                        //                     << std::endl
                        //                     << std::flush;
                        //       });
                        // }
                    }
                    // std::clog << "m2l task(comm(l=" << level << ")) end receive \n";
                }
            }
            catch(std::exception& e)
            {
                std::cerr << " error in task_communication !!!!!!!!!\n";
                std::cerr << " m2l task(comm) crash all level " << level << "  \n" << std::flush;
                if(receive_section > 0)
                {
                    std::cerr << "Bug in receive section \n" << std::flush;
                }
                else
                {
                    std::cerr << "Bug in send section \n" << std::flush;
                }
                std::cerr << e.what() << '\n' << std::flush;
                std::exit(EXIT_FAILURE);
            }
            // #endif
            // std::clog << "m2l task(comm(l=" << level << ")) end\n";
        }   // end task
        // std::cout << "task_communications  fct end\n ";
    }

    /**
     * @brief Perform the communications between the tree_source and the tree_target for the current level
     *
     *  The algorithm is done in three steps
     *   step 1 find the number of data to exchange
     *   step 2 construct the morton index to send
     *   step 3 send the multipoles
     *
     * @tparam TreeS
     * @tparam TreeT
     * @param level  level in the tree
     * @param tree_source  source tree (contains the multipoles)
     * @param tree_target  target tree
     */
    template<typename TreeS, typename TreeT>
    inline auto start_communications(const int& level, TreeS& tree_source, TreeT& tree_target) -> void
    {
        // std::cout << "start_communications  fct start at level " << level << "\n ";

        using mortonIdx_type = std::int64_t;   // typename TreeT::group_of_cell_type::symbolics_type::morton_type;
        // static constexpr int nb_inputs = TreeS::base_type::cell_type::storage_type::inputs_size;
        // static constexpr int dimension = TreeS::base_type::box_type::dimension;

        auto& para = tree_target.get_parallel_manager();
        auto comm = para.get_communicator();

        auto rank = para.get_process_id();
        auto nb_proc = para.get_num_processes();
        if(nb_proc == 1)
        {   // Openmp case -> no communication
            return;
        }
        std::vector<int> nb_messages_to_send(nb_proc, 0);
        std::vector<int> nb_messages_to_receive(nb_proc, 0);

        ///////////////////////////////////////////////////////////////////////////////////
        /// STEP 1
        ///////////////////////////////////////////////////////////////////////////////////
        /// Determines the Morton index vector to be received from processor p. In addition, for each Morton index we
        /// store the cell, i.e. a pointer to its group and the index within the group (group_ptr, index). This will
        /// enable us to insert the multipoles received from processor p directly into the cell.
        ///
        /// to_receive: a vector of vector of pair (the iterator on the group ,the position of the cell in the group)
        ///  to_receive[p] is the position vector of cells in groups whose Morton index comes from processor p
        ///  to_receive[p][i] a pair (the iterator on the group ,the position of the cell in the group)
        /// vector of size nb_proc
        ///    - nb_messages_to_receive: the number of morton indices to  exchange with processor p
        ///    - nb_messages_to_send: the number of morton indices to  send tp processor p
        ///    - morton_to_recv: the morton indices to  exchange with processor p
        /////////////////////////////////////////////////////////////////////////////////
        using grp_access_type = std::pair<decltype(tree_target.begin_cells(level)), int>;
        std::vector<std::vector<grp_access_type>> to_receive(nb_proc);
        std::vector<std::vector<mortonIdx_type>> morton_to_receive(nb_proc);   // TOREMOVE
        bool verbose = false;

        {
            auto begin_left_ghost = tree_source.begin_cells(level);

            auto end_left_ghost = tree_source.begin_mine_cells(level);
            auto begin_right_ghost = tree_source.end_mine_cells(level);
            auto end_right_ghost = tree_source.end_cells(level);
            auto const& distrib = tree_source.get_cell_distribution(level);
            //
            // std::clog << " step 1 M2L" << std::endl << std::flush;

            scalfmm::parallel::comm::start_step1(comm, begin_left_ghost, end_left_ghost, begin_right_ghost,
                                                 end_right_ghost, distrib, nb_messages_to_receive, nb_messages_to_send,
                                                 to_receive, morton_to_receive, verbose);
            // std::cout << " Value after step 1 M2L" << std::endl << std::flush;
            // for(auto p = 0; p < nb_proc; ++p)
            // {
            //     auto& access = to_receive[p];

            //     std::cout << "  to_receive " << p << "  size " << access.size() << std::endl << std::flush;
            //     for(auto i = 0; i < access.size(); ++i)
            //     {
            //         std::cout << i << "  ptr  " << access[i].first->get() << "  index " << access[i].second
            //                   << "  morton "
            //                   << (*(access[i].first))->component(access[i].second).csymbolics().morton_index
            //                   << std::endl;
            //     }
            // }
            tree_source.receive_cells_access(level) = to_receive;
        }
        // for(auto p = 0; p < nb_proc; ++p)
        // {
        //     io::print("   after_step1 morton to receive[" + std::to_string(p) + "] ", morton_to_receive[p]);
        // }
        // io::print("    after_step1 nb_messages_to_receive ", nb_messages_to_receive);
        // io::print("    after_step1 nb_messages_to_send ", nb_messages_to_send);

        ///////////////////////////////////////////////////////////////////////////////////
        /// STEP 2
        ///
        // We can now exchange the morton indices
        // Morton's list of indices to send their multipole to proc p
        // std::vector<std::vector<mortonIdx_type>> morton_to_send(nb_proc);
        // std::clog << " step 2 M2L" << std::endl << std::flush;
        auto& morton_to_send = tree_source.send_morton_indexes(level);
        scalfmm::parallel::comm::start_step2(nb_proc, rank, comm, nb_messages_to_receive, nb_messages_to_send,
                                             morton_to_receive, morton_to_send);

        static constexpr auto prio{omp::priorities::max};
        /////////////////////////////////////////////////////////////////////////////////
        /// STEP 3
        /////////////////////////////////////////////////////////////////////////////////
        /// Processor p sends multipoles to update the multipoles of ghost cells in other processors, so that the
        /// transfer phase can be performed independently (as in OpenMP).
        /// We construct a buffer to pack all the multipoles (first version).
        /// In a second version, to save memory we construct a MPI_TYPE
        /// \warning  We need to make sure that the multipoles are up to date, so we need to put a dependency on the
        /// task.
        ///
        /// First version
        /// This stage is divided into two parts. First, for all the processors I need to communicate with, we pack all
        /// the multipoles to be sent into a buffer and send the buffer. Then we post the receptions to obtain the
        /// multipoles we're going to put in our ghost cells.
        /////////////////////////////////////////////////////////////////////////////////
        // type of dependence
        {
#ifndef __SPLIT_SEND_RECEIV__
            task_communications(level, tree_source, tree_target);   // works
#else
#warning( "segfault in this section ")

            std::clog << " step 3 M2L task_send_multipole_at_level" << std::endl << std::flush;

            // Construct a task to send the multipole
            {
                // loop on the processors to construct the buffer and to send it
                task_send_multipole_at_level(level, tree_source, morton_to_send);
            }

            // Construct a task to receive the multipole
            {
                std::cout << "   task_receive_multipole_at_level start  " << level << std::endl << std::flush;

                task_receive_multipole_at_level(level, tree_source, to_receive);
            }
#endif
        }   // end step3
        // std::cout << "start_communications  fct end at level " << level << "\n ";

    }   // end function start_communications

    template<typename TreeS, typename TreeT>
    inline auto prepare_comm_transfert(const int& level, TreeS& tree_source, TreeT& tree_target) -> void
    {
        // std::cout << "start_communications  fct start at level " << level << "\n ";

        using mortonIdx_type =
          typename TreeS::morton_type;   // typename TreeT::group_of_cell_type::symbolics_type::morton_type;
        // static constexpr int nb_inputs = TreeS::base_type::cell_type::storage_type::inputs_size;
        // static constexpr int dimension = TreeS::base_type::box_type::dimension;

        auto& para = tree_source.get_parallel_manager();
        auto comm = para.get_communicator();

        auto rank = para.get_process_id();
        auto nb_proc = para.get_num_processes();
        if(nb_proc == 1)
        {   // Openmp case -> no communication
            return;
        }
        std::vector<int> nb_messages_to_send(nb_proc, 0);
        std::vector<int> nb_messages_to_receive(nb_proc, 0);

        ///////////////////////////////////////////////////////////////////////////////////
        /// STEP 1
        ///////////////////////////////////////////////////////////////////////////////////
        /// Determines the Morton index vector to be received from processor p. In addition, for each Morton index we
        /// store the cell, i.e. a pointer to its group and the index within the group (group_ptr, index). This will
        /// enable us to insert the multipoles received from processor p directly into the cell.
        ///
        /// to_receive: a vector of vector of pair (the iterator on the group ,the position of the cell in the group)
        ///  to_receive[p] is the position vector of cells in groups whose Morton index comes from processor p
        ///  to_receive[p][i] a pair (the iterator on the group ,the position of the cell in the group)
        /// vector of size nb_proc
        ///    - nb_messages_to_receive: the number of morton indices to  exchange with processor p
        ///    - nb_messages_to_send: the number of morton indices to  send tp processor p
        ///    - morton_to_recv: the morton indices to  exchange with processor p
        /////////////////////////////////////////////////////////////////////////////////
        using grp_access_type = std::pair<decltype(tree_target.begin_cells(level)), int>;
        std::vector<std::vector<grp_access_type>> to_receive(nb_proc);
        std::vector<std::vector<mortonIdx_type>> morton_to_receive(nb_proc);   // TOREMOVE
        bool verbose = false;

        {
            auto begin_left_ghost = tree_source.begin_cells(level);

            auto end_left_ghost = tree_source.begin_mine_cells(level);
            auto begin_right_ghost = tree_source.end_mine_cells(level);
            auto end_right_ghost = tree_source.end_cells(level);
            auto const& distrib = tree_source.get_cell_distribution(level);
            //
            // std::clog << " step 1 M2L" << std::endl << std::flush;

            scalfmm::parallel::comm::start_step1(comm, begin_left_ghost, end_left_ghost, begin_right_ghost,
                                                 end_right_ghost, distrib, nb_messages_to_receive, nb_messages_to_send,
                                                 to_receive, morton_to_receive, verbose);
            // std::cout << " Value after step 1 M2L" << std::endl << std::flush;
            // for(auto p = 0; p < nb_proc; ++p)
            // {
            //     auto& access = to_receive[p];

            //     std::cout << "  to_receive " << p << "  size " << access.size() << std::endl << std::flush;
            //     for(auto i = 0; i < access.size(); ++i)
            //     {
            //         std::cout << i << "  ptr  " << access[i].first->get() << "  index " << access[i].second
            //                   << "  morton "
            //                   << (*(access[i].first))->component(access[i].second).csymbolics().morton_index
            //                   << std::endl;
            //     }
            // }
            tree_source.receive_cells_access(level) = to_receive;
        }
        // for(auto p = 0; p < nb_proc; ++p)
        // {
        //     io::print("   after_step1 morton to receive[" + std::to_string(p) + "] ", morton_to_receive[p]);
        // }
        // io::print("    after_step1 nb_messages_to_receive ", nb_messages_to_receive);
        // io::print("    after_step1 nb_messages_to_send ", nb_messages_to_send);

        ///////////////////////////////////////////////////////////////////////////////////
        /// STEP 2
        ///
        // We can now exchange the morton indices
        // Morton's list of indices to send their multipole to proc p
        // std::vector<std::vector<mortonIdx_type>> morton_to_send(nb_proc);
        // std::clog << " step 2 M2L" << std::endl << std::flush;
        auto& morton_to_send = tree_source.send_morton_indexes(level);
        scalfmm::parallel::comm::start_step2(nb_proc, rank, comm, nb_messages_to_receive, nb_messages_to_send,
                                             morton_to_receive, morton_to_send);
    }

    ///////////////////////////////////////////////////////////////////////////////////

    /// @brief apply the transfer operator to construct the local approximation in tree_target
    ///
    /// @tparam TreeS template for the Tree source type
    /// @tparam TreeT template for the Tree target type
    /// @tparam FarField template for the far field type
    /// @tparam BufferPtr template for the type of pointer of the buffer
    /// @param tree_source the tree containing the source cells/leaves
    /// @param tree_target the tree containing the target cells/leaves
    /// @param far_field The far field operator
    /// @param buffers vector of buffers used by the far_field in the transfer pass (if needed)
    /// @param split the enum  (@see split_m2l) tp specify on which level we apply the transfer
    /// operator
    ///
    template<typename TreeS, typename TreeT, typename FarField, typename BufferPtr>
    inline auto transfer(TreeS& tree_source, TreeT& tree_target, FarField const& far_field,
                         std::vector<BufferPtr> const& buffers,
                         omp::pass::split_m2l split = omp::pass::split_m2l::full) -> void
    {
        //
        auto tree_height = tree_target.height();

        /////////////////////////////////////////////////////////////////////////////////////////
        ///  Loop on the level from the level top_height to the leaf level (Top to Bottom)
        auto top_height = tree_target.box().is_periodic() ? 1 : 2;
        auto last_level = tree_height;

        switch(split)
        {
        case omp::pass::split_m2l::remove_leaf_level:
            last_level--;
            ;
            break;
        case omp::pass::split_m2l::leaf_level:
            top_height = (tree_height - 1);
            break;
        case omp::pass::split_m2l::full:
            break;
        }
        // std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;

        // std::cout << "  loop start at top_height " << top_height << " and end at last_level " << last_level << std::endl
        //           << std::flush;
        for(std::size_t level = top_height; level < last_level; ++level)
        {
            // std::cout << "transfer  : " << level << std::endl << std::flush;
            start_communications(level, tree_source, tree_target);
            // std::cout << "   end comm  " << level << std::endl << std::flush;
            // #pragma omp taskwait
            // std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;

            omp::pass::transfer_level(level, tree_source, tree_target, far_field, buffers);
            //             std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
        }
        // std::cout << "end transfer pass" << std::endl << std::flush;
        // std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
    }

}   // namespace scalfmm::algorithms::mpi::pass
#endif   //_OPENMP
#endif   // SCALFMM_ALGORITHMS_MPI_TRANSFER_HPP
