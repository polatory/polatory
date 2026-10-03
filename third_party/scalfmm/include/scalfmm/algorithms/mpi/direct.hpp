// --------------------------------
// See LICENCE file at project root
// File : algorithm/mpi/direct.hpp
// --------------------------------
#ifndef SCALFMM_ALGORITHMS_MPI_DIRECT_HPP
#define SCALFMM_ALGORITHMS_MPI_DIRECT_HPP

#include "scalfmm/algorithms/omp/direct.hpp"
#include "scalfmm/parallel/mpi/comm.hpp"

#include <mpi.h>

namespace scalfmm::algorithms::mpi::pass
{
    namespace comm
    {
        /**
	* @brief Perform the communications between the tree_source and the tree_target for the current level
	*
	*  THE algorithm is done in three steps
	*   step 1 find the number of data to exchange
	*   step 2 construct the morton index to send
	*   step 3 send/receive the particles
	*
	* @tparam TreeS
    *
	* @param tree_source  source tree (contains the particles)
	*/
        template<typename TreeS>
        inline auto start_communications(TreeS& tree_source) -> void
        {
            auto& para = tree_source.get_parallel_manager();
            auto comm = para.get_communicator();
            auto rank = para.get_process_id();
            auto nb_proc = para.get_num_processes();
            if(nb_proc > 1)
            {   // Openmp case -> no communication

                //
                using grp_access_type = std::pair<decltype(tree_source.begin_leaves()), int>;
                using mortonIdx_type = std::int64_t;
                //
                std::vector<std::vector<grp_access_type>> leaf_to_receive_access(nb_proc);
                std::vector<std::vector<grp_access_type>> leaf_to_send_access(nb_proc);
                std::vector<std::vector<mortonIdx_type>> morton_to_receive(nb_proc);   // TOREMOVE
                std::vector<int> nb_messages_to_send(nb_proc, 0);
                std::vector<int> nb_messages_to_receive(nb_proc, 0);
                ///
                auto begin_left_ghost = tree_source.begin_leaves();
                auto end_left_ghost = tree_source.begin_mine_leaves();
                auto begin_right_ghost = tree_source.end_mine_leaves();
                auto end_right_ghost = tree_source.end_leaves();
                //
                //print leaf block
                // for(auto it = end_left_ghost; it != begin_right_ghost; ++it)
                // {
                //     std::cout << **it << std::endl;
                // }
                //
                auto const& leaf_distribution = tree_source.get_leaf_distribution();

                scalfmm::parallel::comm::start_step1(comm, begin_left_ghost, end_left_ghost, begin_right_ghost,
                                                     end_right_ghost, leaf_distribution, nb_messages_to_receive,
                                                     nb_messages_to_send, leaf_to_receive_access, morton_to_receive);
                // for(auto p = 0; p < nb_proc; ++p)
                // {
                //     io::print("    morton to receive[" + std::to_string(p) + "] ", morton_to_receive[p]);
                // }
                // io::print("    nb_messages_to_receive ", nb_messages_to_receive);
                // io::print("    nb_messages_to_send ", nb_messages_to_send);
                //
                std::vector<std::vector<mortonIdx_type>> morton_to_send(nb_proc);
                //
                scalfmm::parallel::comm::start_step2(nb_proc, rank, comm, nb_messages_to_receive, nb_messages_to_send,
                                                     morton_to_receive, morton_to_send);

                /////////////////////////////////////////////////////////////////////////////////
                /// STEP 3
                /////////////////////////////////////////////////////////////////////////////////
                // send the particles
                // morton_to_send list des indices de Morton.
                // leaf_to_send_access (ptr on the group and index into  the group)
                //
                auto begin_grp = tree_source.begin_mine_leaves();
                auto end_grp = tree_source.end_mine_leaves();

                scalfmm::parallel::comm::build_direct_access_to_components(nb_proc, begin_grp, end_grp,
                                                                           leaf_to_send_access, morton_to_send);
                //
                // Build the mpi type for the particles
                //
                static constexpr std::size_t dimension = TreeS::base_type::leaf_type::dimension;
                static constexpr std::size_t inputs_size = TreeS::base_type::leaf_type::inputs_size;

                using position_coord_type = typename TreeS::base_type::leaf_type::position_coord_type;
                using inputs_type_ori = typename TreeS::base_type::leaf_type::inputs_type;

                static_assert(!meta::is_complex_v<inputs_type_ori>, "input complex type not yet supported.");
                using inputs_type1 = inputs_type_ori;
                using inputs_type = std::conditional_t<meta::is_complex_v<inputs_type_ori>,
                                                       meta::has_value_type_t<inputs_type_ori>, inputs_type_ori>;
                // for complex value (2) otherwise 1  NOT YET USED for particles
                int nb_input_values = meta::is_complex_v<inputs_type_ori> ? 2 : 1;

                auto mpi_position_type = cpp_tools::parallel_manager::mpi::get_datatype<position_coord_type>();
                auto mpi_input_type = cpp_tools::parallel_manager::mpi::get_datatype<inputs_type>();
                //

                // build and commit the MPI type of the particle to send
                // std::cout << "=================== Send type ========================\n";

                auto particle_type_to_send = scalfmm::parallel::comm::build_mpi_particles_type<dimension>(
                  leaf_to_send_access, inputs_size, mpi_position_type, mpi_input_type);

                // send the particles
                for(auto p = 0; p < nb_proc; ++p)
                {
                    if(leaf_to_send_access[p].size() != 0)
                    {
                        comm.isend(MPI_BOTTOM, 1, particle_type_to_send[p], p, 777);
                    }
                }
                //
                // receive the particle
                std::vector<cpp_tools::parallel_manager::mpi::request> recept_mpi_status;
                // build and commit the MPI type of the particle to receive
                // std::cout << "=================== Receive type ========================\n";

                auto particle_type_to_receive = scalfmm::parallel::comm::build_mpi_particles_type<dimension>(
                  leaf_to_receive_access, inputs_size, mpi_position_type, mpi_input_type);

                for(auto p = 0; p < nb_proc; ++p)
                {
                    if(leaf_to_receive_access[p].size() != 0)
                    {
                        recept_mpi_status.push_back(comm.irecv(MPI_BOTTOM, 1, particle_type_to_receive[p], p, 777));
                    }
                }
                if(recept_mpi_status.size() > 0)
                {
                    cpp_tools::parallel_manager::mpi::request::waitall(recept_mpi_status.size(),
                                                                       recept_mpi_status.data());
                }

                //print leaf block
                // std::cout << "==========================================================\n";
                // int id_group{0};
                // for(auto ptr_group = begin_left_ghost; ptr_group != end_right_ghost; ++ptr_group)
                // {
                //     auto const& current_group_symbolics = (*ptr_group)->csymbolics();

                //     std::cout << "*** Group of leaf index " << ++id_group << " *** index in ["
                //               << current_group_symbolics.starting_index << ", " << current_group_symbolics.ending_index
                //               << "[";
                //     std::cout << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                //     std::cout << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
                //     std::cout << "global index =  " << current_group_symbolics.idx_global << " \n";
                //     std::cout << "    index: ";
                //     (*ptr_group)->cstorage().print_block_data(std::cout);
                // }
            }

            // #ifndef _DEBUG_BLOCK_DATA
            //             std::clog << "  FINAl block\n";
            //             int id_group{0};
            //             auto group_of_leaves = tree_source.vector_of_leaf_groups();
            //             for(auto pg: group_of_leaves)
            //             {
            //                 auto const& current_group_symbolics = pg->csymbolics();
            //                 std::cout << "*** Group of leaf index " << ++id_group << " *** index in ["
            //                           << current_group_symbolics.starting_index << ", " << current_group_symbolics.ending_index
            //                           << "[";
            //                 std::cout << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
            //                 std::cout << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
            //                 std::cout << "global index =  " << current_group_symbolics.idx_global << " \n" << std::flush;
            //                 // std::cout << "    index: ";
            //                 // std::clog << "block index " << tt++ << std::endl;
            //                 pg->cstorage().print_block_data(std::clog);
            //             }
            //             std::clog << "  ---------------------------------------------------\n";
            // #endif
        }   // end start_communications
    }   // namespace comm

    // template<typename Tree, typename NearField>
    // inline auto direct_mine_ghost(Tree const& tree, NearField const& nearfield) -> void
    // {
    // }
    /**
     * @brief Compute direct interaction between particles
     *
     *  When tree_source = tree_target we call the direct(tree_source, nearfield) function
     *
     * @tparam TreeSource
     * @tparam TreeTarget
     * @tparam NearField
     *
     * @param tree_source the source tree
     * @param tree_target  tke target tree where we compute the field
     * @param nearfield  the near-field operator
     */
    template<typename TreeSource, typename TreeTarget, typename NearField>
    inline auto direct(TreeSource& tree_source, TreeTarget& tree_target, NearField const& near_field) -> void
    {
        // std::cout << "direct  : " << std::endl << std::flush;
        comm::start_communications(tree_source);
        // std::cout << "   end comm  " << std::endl << std::flush;
        // std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
        omp::pass::direct(tree_source, tree_target, near_field);
        // if(nearfield.mutual())
        // {
        //     direct_mine_ghost(tree_target, near_field);
        // }
    }
}   // namespace scalfmm::algorithms::mpi::pass

#endif
