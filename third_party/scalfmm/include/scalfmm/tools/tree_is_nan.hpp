#pragma once

#include <array>
#include <fstream>
#include <string>

#include "scalfmm/tree/box.hpp"
// #include "scalfmm/tree/dist_group_tree.hpp"
#include "scalfmm/tree/group_tree_view.hpp"
#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"
#include <cpp_tools/parallel_manager/parallel_manager.hpp>

#include <cpp_tools/colors/colorized.hpp>

namespace scalfmm::tools::io
{
    /////////////////////////////////
    ///
    /**
     * @brief check if there is any NaN in the multipoles are the locals of a tree
     *
     * For all levels, depending on the option, we check whether the multipole and local tensors contain NaN values. The
     * output (error output) is colored for ease of reading.
     *
     * option 1 only the multipoles
     * option 2 only the locals
     * otherwise both multipoles and locals
     *
     * @param tree the tree (tree view) to check
     * @param option int (1,2 ... ) - the option described above
     * @return the test value
     */
    template<typename TreeType>
    inline auto check_nan_in_tree(TreeType const& tree, int option = -1) -> bool
    {
        bool check_mult{true}, check_loc{true};

        // loop through all the levels of the tree
        for(std::size_t level = tree.leaf_level(); level >= tree.top_level(); --level)
        {
            std::cout << cpp_tools::colors::magenta << "\n\tLEVEL = " << level << cpp_tools::colors::reset << std::endl;
            auto start = tree.begin_mine_cells(level);

            // loop through all the group of the current level
            for(auto grp = start; grp != tree.end_mine_cells(level); ++grp)
            {
                // loop over all the cells of the current group
                for(std::size_t index = 0; index < (*grp)->size(); ++index)
                {
                    auto const& cell = (*grp)->ccomponent(index);

                    // multipoles
                    if(option != 2)
                    {
                        auto mults = cell.cmultipoles();
                        auto number_of_arrays = mults.size();
                        for(std::size_t l = 0; l < number_of_arrays; ++l)
                        {
                            auto& mult = mults.at(l);
                            std::cerr << "level " << level << " - Cell morton " << cell.csymbolics().morton_index
                                      << std::endl;
                            if(xt::any(xt::isnan(mult)))
                            {
                                check_mult = false;
                                std::cerr << cpp_tools::colors::red;
                            }
                            else
                            {
                                std::cerr << cpp_tools::colors::green;
                            }
                            std::cerr << "mult(" << l << ")\n" << mult << std::endl;
                            std::cerr << cpp_tools::colors::reset;
                        }
                    }

                    // locals
                    if(option != 1)
                    {
                        auto const& locals = cell.clocals();
                        auto number_of_arrays = locals.size();
                        for(std::size_t l = 0; l < number_of_arrays; ++l)
                        {
                            auto const& local = locals.at(l);
                            std::cerr << "level " << level << " - Cell morton " << cell.csymbolics().morton_index
                                      << std::endl;
                            if(xt::any(xt::isnan(local)))
                            {
                                check_loc = false;
                                std::cerr << cpp_tools::colors::red;
                            }
                            else
                            {
                                std::cerr << cpp_tools::colors::cyan;
                            }
                            std::cerr << "local(" << l << ")\n" << local << std::endl;
                            std::cerr << cpp_tools::colors::reset;
                        }
                    }
                }
            }
        }

        return check_mult && check_loc;
    }
};   // namespace scalfmm::tools::io
