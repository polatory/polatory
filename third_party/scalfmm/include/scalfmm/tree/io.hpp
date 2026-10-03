// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tree/io.hpp
// --------------------------------
#pragma once

#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace scalfmm::io
{
    /**
     * @brief
     *
     * @tparam CellType
     * @param cell
     * @param print_aux
     */
    template<typename CellType>
    auto print_cell(const CellType& cell, bool print_aux = false) -> void
    {
        auto m = cell.cmultipoles();

        auto nb_m = m.size();
        std::cout << "cell index: " << cell.index() << " level " << cell.csymbolics().level << "\n";
        for(std::size_t i{0}; i < nb_m; ++i)
        {
            auto ten = m.at(i);
            std::cout << "  multi(" << i << "): \n" << m.at(i) << std::endl;
        }
        auto loc = cell.clocals();
        auto nb_loc = loc.size();
        for(std::size_t i{0}; i < nb_loc; ++i)
        {
            auto ten = loc.at(i);
            std::cout << &(loc.at(i)) << std::endl;
            std::cout << "  loc(" << i << "): \n" << loc.at(i) << std::endl;
        }
    }

    /**
     * @brief  Display particles in leaf and compute the current Morton index of the particle
     *
     * @tparam LeafType type
     * @tparam Box type
     * @param leaf the current leaf
     * @param box  The box to compute the Morton index
     * @param level  The level to compute the Morton index
     */
    template<typename LeafType, typename Box>
    auto print_leaf(const LeafType& leaf, Box box, int level) -> void
    {
        auto morton = leaf.index();
        int i{0};
        for(auto const& pl: leaf)
        {
            const auto p = typename LeafType::const_proxy_type(pl);
            std::cout << i++ << " morton: " << morton << " part= " << p << " cmp_morton "
                      << scalfmm::index::get_morton_index(p.position(), box, level) << std::endl;
        }
    }

    /**
     * @brief Display particles in leaf
     *
     * @tparam LeafType the type of the leaf
     * @param leaf the current leaf
     */
    template<typename LeafType>
    auto print_leaf(const LeafType& leaf) -> void
    {
        auto morton = leaf.index();
        int i{0};
        for(auto const& p: leaf)
        {
            const auto pp = typename LeafType::const_proxy_type(p);
            std::cout << i++ << " morton: " << morton << " part= " << pp << std::endl;
        }
    }

    /**
     * @brief trace the index of the cells and leaves in the tree
     *
     * Depending on the level we print more or less details
     *  level_trace = 1 print minimal information (height, order, group size)
     *  level_trace = 2 print information of the tree (group interval and index inside)
     *  level_trace = 3 print information of the tree (leaf interval and index inside and their p2p interaction
     *  list) level_trace = 4 print information of the tree (cell interval and index inside and their m2l
     *  interaction list)
     * level_trace = 5 print information of the tree (leaf and cell interval and index inside
     *  and their p2p and m2l interaction lists)
     *
     * @warning to have the right p2p list we have to have the group size in the tree equal to the number
     *  of leaves otherwise, we only print the index inside the group.
     *
     * @tparam TreeType
     * @param os
     * @param tree
     * @param level_trace
     */
    template<typename TreeType>
    inline auto trace(std::ostream& os, const TreeType& tree, const std::size_t level_trace = 0) -> void
    {
        std::cout << "Trace of the tree\n";

        auto level_0 = []() {};

        auto level_1 = [&tree, &os]()
        {
            os << "group_tree | height = " << tree.height() << '\n' << std::flush;
            os << "group_tree | order =  " << tree.order() << '\n' << std::flush;
            os << "group_tree | Blocking group size for leaves = " << tree.group_of_leaf_size() << '\n' << std::flush;
            os << "group_tree | Blocking group size for cells =  " << tree.group_of_cell_size() << '\n' << std::flush;
            os << "group_tree | number of leaves group =         " << tree.leaf_groups_size() << '\n' << std::flush;
            auto cell_level_it = tree.cbegin_cells() + (tree.height() - 1);

            int top_level = tree.box().is_periodic() ? 0 : 2;
            for(int level = int(tree.height()) - 1; level >= top_level; --level, --cell_level_it)
            {
                auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                auto group_of_cell_end = std::cend(*(cell_level_it));
                os << "group_tree | number of cells group (" << level
                   << ")= " << std::distance(group_of_cell_begin, group_of_cell_end) << '\n'
                   << std::flush;
            }
        };

        auto level_2 = [&tree, &os]()
        {
            auto tree_height = tree.height();
            std::size_t id_group{0};
            os << "======================================================================\n" << std::flush;
            os << "========== leaf level : " << tree_height - 1 << " ============================\n" << std::flush;
            os << tree.group_of_leaf_size() << " groups at leaf level.\n" << std::flush;

            std::for_each(tree.cbegin_leaves(), tree.cend_leaves(),
                          //    std::cbegin(m_group_of_leaf), std::cend(m_group_of_leaf),
                          [&id_group, &os](auto const& ptr_group)
                          {
                              auto const& current_group_symbolics = ptr_group->csymbolics();

                              os << "*** Group of leaf index " << ++id_group << " *** index in ["
                                 << current_group_symbolics.starting_index << ", "
                                 << current_group_symbolics.ending_index << "[";
                              os << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                              os << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
                              os << "global index =  " << current_group_symbolics.idx_global << " \n";
                              os << "    index: ";
                              component::for_each(std::begin(*ptr_group), std::end(*ptr_group), [&os](auto& leaf)
                                                  { os << leaf.index() << "(" << leaf.size() << ") "; });
                              os << std::endl;
                          });
            os << "======================================================================\n" << std::flush;
            os << "======================================================================\n" << std::flush;

            //    auto cell_level_it = std::cbegin(m_group_of_cell_per_level) + (tree_height - 1);
            auto cell_level_it = tree.cbegin_cells() + (tree_height - 1);

            id_group = 0;
            int top_level = tree.box().is_periodic() ? 0 : 2;
            for(int level = int(tree_height) - 1; level >= top_level; --level)
            {
                os << "========== level : " << level << " ============================\n";
                // auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                // auto group_of_cell_end = std::cend(*(cell_level_it));
                auto group_of_cell_begin = tree.begin_cells(level);
                auto group_of_cell_end = tree.end_cells(level);
                std::for_each(group_of_cell_begin, group_of_cell_end,
                              [&id_group, &os](auto const& ptr_group)
                              {
                                  auto const& current_group_symbolics = ptr_group->csymbolics();
                                  os << "*** Group of cell index " << ++id_group << " *** index in ["
                                     << current_group_symbolics.starting_index << ", "
                                     << current_group_symbolics.ending_index << "[";
                                  os << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                                  os << "    group size:  " << current_group_symbolics.number_of_component_in_group
                                     << ", ";
                                  os << "global index =  " << current_group_symbolics.idx_global << " \n"
                                     << " ref: depend(multi)=" << &ptr_group->ccomponent(0).cmultipoles(0)
                                     << " rf depend(locals)=" << &ptr_group->ccomponent(0).clocals(0) << " \n";
                                  os << "    index: ";
                                  component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                                      [&os](auto& cell) { os << cell.index() << " "; });
                                  os << std::endl;
                              });
                --cell_level_it;
            }
        };
        auto level_leaf_p2p = [&tree, &os]()
        {
            std::size_t id_group{0};
            os << "========================== P2P in group interactions list ========================= \n";
            const auto& group_of_leaves = tree.vector_of_leaf_groups();

            os << group_of_leaves.size() << " groups at leaf level.\n";
            std::for_each(tree.cbegin_leaves(), tree.cend_leaves(),
                          //                 std::cbegin(group_of_leaves), std::cend(group_of_leaves),
                          [&id_group, &os](auto const& ptr_group)
                          {
                              auto const& current_group_symbolics = ptr_group->csymbolics();

                              os << "*** Group of leaf index " << ++id_group << " *** index in ["
                                 << current_group_symbolics.starting_index << ", "
                                 << current_group_symbolics.ending_index << "[";
                              os << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                              os << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
                              os << "global index =  " << current_group_symbolics.idx_global << " \n";
                              os << "    index: \n";
                              int cpti = 0;

                              component::for_each(
                                std::begin(*ptr_group), std::end(*ptr_group),
                                [&cpti, &os](auto& leaf)
                                {
                                    auto& leaf_symbolics = leaf.symbolics();
                                    os << "       " << cpti++ << "  " << leaf.index() << "  p2p_list ("
                                       << leaf_symbolics.existing_neighbors_in_group << "): ";
                                    // get the p2p interaction list
                                    // auto index = leaf_symbolics.interaction_indexes;
                                    auto index = leaf_symbolics.interaction_iterators;
                                    for(std::size_t idx = 0; idx < leaf_symbolics.existing_neighbors_in_group; ++idx)
                                    {
                                        os << index[idx]->index() << " ";
                                    }
                                    os << std::endl;
                                });
                              cpti = 0;
                              os << "     Number of out of group interactions "
                                 << current_group_symbolics.outside_interactions.size() << std::endl;
                              for(auto u: current_group_symbolics.outside_interactions)
                              {
                                  os << "       " << cpti++ << "  " << u << std::endl;
                              }
                              os << std::endl;
                          });
        };
        auto level_leaf_m2l = [&tree, &os]()
        {
            auto tree_height = tree.height();

            std::size_t id_group{0};
            os << "========================== M2L interaction list ========================= \n" << std::flush;

            auto cell_level_it = tree.cbegin_cells() + (tree_height - 1);
            id_group = 0;
            int top_level = tree.box().is_periodic() ? 0 : 2;
            for(int level = int(tree_height) - 1; level >= top_level; --level)
            {
                os << "========== level : " << level << " ============================\n" << std::flush;
                auto group_of_cell_begin = std::cbegin(*(cell_level_it));
                auto group_of_cell_end = std::cend(*(cell_level_it));
                std::for_each(
                  group_of_cell_begin, group_of_cell_end,
                  [&id_group, &os](auto const& ptr_group)
                  {
                      auto const& current_group_symbolics = ptr_group->csymbolics();
                      os << "*** Group of cell index " << ++id_group << " *** index in ["
                         << current_group_symbolics.starting_index << ", " << current_group_symbolics.ending_index
                         << "[" << std::flush;
                      os << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n" << std::flush;
                      os << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", "
                         << std::flush;
                      os << "global index =  " << current_group_symbolics.idx_global << " \n"
                         << " ref: depend(multi)=" << &ptr_group->ccomponent(0).cmultipoles(0)
                         << " rf depend(locals)=" << &ptr_group->ccomponent(0).clocals(0) << " \n"
                         << std::flush;
                      os << "    index: \n" << std::flush;
                      int cpt = 0;
                      component::for_each(std::begin(*ptr_group), std::end(*ptr_group),
                                          [&cpt, &os](auto& cell)
                                          {
                                              auto& cell_symbolics = cell.symbolics();
                                              os << "       " << cpt++ << "  " << cell.index() << "  m2l_list ("
                                                 << cell_symbolics.existing_neighbors << "): " << std::flush;
                                              // get the m2l interaction list
                                              auto index = cell_symbolics.interaction_iterators;
                                              for(std::size_t idx = 0; idx < cell_symbolics.existing_neighbors; ++idx)
                                              {
                                                  os << index[idx]->index() << " ";
                                              }
                                              os << std::endl << std::flush;
                                          });
                      os << std::endl;
                  });
                --cell_level_it;
            }
        };
        switch(level_trace)
        {
        case 0:
            level_0();
            break;
        case 1:
            level_1();
            break;
        case 2:
            level_2();
            break;
        case 3:
            level_1();
            level_leaf_p2p();
            break;
        case 4:
            level_1();
            level_leaf_m2l();
            break;
        case 5:
            level_1();
            level_leaf_p2p();
            level_leaf_m2l();
            break;

        default:
            level_0();
        }
    }

    /**
     * @brief trace the group dependencies for the transfer pass (M2L)
     *
     * @tparam TreeType
     * @param os
     * @param tree
     */
    template<typename TreeType>
    inline auto trace_m2l_dep(std::ostream& os, const TreeType& tree) -> void
    {
        std::cout << "Trace of the m2l group dependencies\n";
        auto tree_height = tree.height();

        std::size_t id_group{0};
        os << "========================== M2L Group dependencies ========================= \n";

        id_group = 0;
        int top_level = tree.box().is_periodic() ? 0 : 2;
        for(int level = int(tree_height) - 1; level >= top_level; --level)
        {
            os << "========== level : " << level << " ============================\n";
            const auto group_of_cell_begin = tree.cbegin_mine_cells(level);
            const auto group_of_cell_end = tree.cend_mine_cells(level);
            std::for_each(group_of_cell_begin, group_of_cell_end,
                          [&id_group, &os](auto const& ptr_group)
                          {
                              auto const& current_group_symbolics = ptr_group->csymbolics();
                              os << "*** Group of cell index " << ++id_group << " *** index in ["
                                 << current_group_symbolics.starting_index << ", "
                                 << current_group_symbolics.ending_index << "[";
                              os << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                              os << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
                              os << "global index =  " << current_group_symbolics.idx_global << " \n";

                              io::print(" m2l depend(in) ", current_group_symbolics.group_dependencies_m2l);
                              std::cout << " m2l depend(out) " << &ptr_group->ccomponent(0).clocals(0) << " \n";
                          });
        }
    }

    /**
     * @brief
     *
     */
    auto print_map = [](auto const comment, auto const& map)
    {
        std::cout << comment << "{";
        for(const auto& pair: map)
            std::cout << "{" << pair.first << ": " << pair.second << "}";
        std::cout << "}\n";
    };

    /**
     * @brief
     *
     */
    auto print_with_map = [](std::ostream& os, auto const comment, auto const& v, auto& map)
    {
        std::cout << comment << " (" << v.size() << ") ";
        if(v.size() > 0)
        {
            for(auto& i: v)
            {
                std::ostringstream address;
                address << static_cast<void const*>(i);
                std::cout << map[address.str()] << " ";
                // std::cout << "( " << i << ", " << key_loc[address.str()] << ") ";
            }
        }
        os << "\n";
    };

    /**
     * @brief
     *
     * @tparam TreeType
     * @param tree
     */
    template<typename TreeType>
    inline auto init_trace_group_dependencies(TreeType& tree) -> void
    {
        std::size_t id_group{0};
        const std::size_t top_level = tree.box().is_periodic() ? 0 : 2;

        for(std::size_t level = top_level; level < tree.height(); ++level)
        {
            std::cout << "Level: " << level << " start " << id_group;
            const auto group_of_cell_begin = tree.cbegin_mine_cells(level);
            const auto group_of_cell_end = tree.cend_mine_cells(level);
            std::for_each(group_of_cell_begin, group_of_cell_end,
                          [&id_group, &tree](auto const& ptr_group)
                          {
                              std::ostringstream address_mult, address_loc;

                              auto& current_group_symbolics = ptr_group->symbolics();
                              current_group_symbolics.idx_global_tree = id_group;
                              auto ptr_mult = &(ptr_group->ccomponent(0).cmultipoles(0));
                              address_mult << (void const*)ptr_mult;
                              address_loc << static_cast<void const*>(&(ptr_group->ccomponent(0).clocals(0)));
                              //   const std::string m(address_mult.str()), l(address_loc.str());
                              tree.m_multipoles_dependencies[address_mult.str()] = id_group;
                              tree.m_locals_dependencies[address_loc.str()] = id_group;
                              ++id_group;
                          });
            std::cout << " end " << id_group - 1 << std::endl;
        }
        print_map("Map mult: ", tree.m_multipoles_dependencies);
        print_map("Map loc: ", tree.m_locals_dependencies);
    }

    /**
     * @brief
     *
     * @tparam SourceTreeType
     * @tparam TargetTreeType
     * @param os
     * @param source_tree
     * @param target_tree
     */
    template<typename SourceTreeType, typename TargetTreeType>
    inline auto trace_group_dependencies(std::ostream& os, SourceTreeType& source_tree, TargetTreeType& target_tree)
      -> void
    {
        std::cout << "Trace of the  group dependencies\n";
        bool same_tree{false};
        if constexpr(std::is_same_v<SourceTreeType, TargetTreeType>)
        {
            same_tree = (&source_tree == &target_tree);
        }
        auto& tree = target_tree;
        init_trace_group_dependencies(source_tree);
        init_trace_group_dependencies(target_tree);

        os << "========================== M2L Group dependencies ========================= \n";

        const std::size_t top_level = tree.box().is_periodic() ? 0 : 2;
        const auto leaf_level = tree.leaf_level();
        auto& key_mul = source_tree.m_multipoles_dependencies;
        auto& key_loc = target_tree.m_locals_dependencies;
        os << "======================================================\n";
        os << "==========       M2M dependencies  source       ======\n";
        os << "======================================================\n";
        for(std::size_t level = leaf_level - 1; level >= top_level; --level)
        {
            os << "========== level : " << level << " ============================\n";
            const auto group_of_cell_begin = source_tree.cbegin_mine_cells(level);
            const auto group_of_cell_end = source_tree.cend_mine_cells(level);
            std::for_each(
              group_of_cell_begin, group_of_cell_end,
              [&os, &key_mul](auto const& ptr_group)
              {
                  auto const& current_group_symbolics = ptr_group->csymbolics();
                  os << "*** Group of cell index " << current_group_symbolics.idx_global_tree << " *** index in ["
                     << current_group_symbolics.starting_index << ", " << current_group_symbolics.ending_index << "[";
                  //   os << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                  //   os << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
                  os << "global index (level)=  " << current_group_symbolics.idx_global << " \n";

                  print_with_map(os, " m2m depend(in) ", current_group_symbolics.group_dependencies_m2m_in, key_mul);
                  //   os << std::endl;
              });
        }
        os << "======================================================\n";
        os << "==========       target dependencies            ======\n";
        os << "======================================================\n";
        // for(std::size_t level = leaf_level; level >= top_level; --level)
        for(std::size_t level = leaf_level; level >= top_level; --level)
        {
            os << "========== level : " << level << " ============================\n";
            const auto group_of_cell_begin = target_tree.cbegin_mine_cells(level);
            const auto group_of_cell_end = target_tree.cend_mine_cells(level);
            std::for_each(
              group_of_cell_begin, group_of_cell_end,
              [&os, level, top_level, &key_mul, &key_loc](auto const& ptr_group)
              {
                  auto const& current_group_symbolics = ptr_group->csymbolics();
                  os << "*** Group of cell index " << current_group_symbolics.idx_global_tree << " *** index in ["
                     << current_group_symbolics.starting_index << ", " << current_group_symbolics.ending_index << "[";
                  os << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                  //   os << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
                  //   os << "global index (level)=  " << current_group_symbolics.idx_global << " \n";

                  //   std::ostringstream address_mult, address_loc;
                  //   address_mult << static_cast<void const*>(&(ptr_group->ccomponent(0).cmultipoles(0)));
                  //   address_loc << static_cast<void const*>(&(ptr_group->ccomponent(0).clocals(0)));

                  //   std::cout << " depend on multipoles: " << &(ptr_group->ccomponent(0).cmultipoles(0)) << " grp "
                  //             << key_mul[address_mult.str()] << " \n"
                  //             << " depend on locals:      " << &(ptr_group->ccomponent(0).clocals(0)) << " grp "
                  //             << key_loc[address_loc.str()] << " \n";

                  print_with_map(os, " m2l depend(in) ", current_group_symbolics.group_dependencies_m2l, key_mul);

                  if(level != top_level)
                  {
                      print_with_map(os, " l2l depend(in) ", current_group_symbolics.group_dependencies_l2l_in,
                                     key_loc);
                  }
                  os << std::endl;
              });
        }
    }

    /**
     * @brief
     *
     * @tparam TreeType
     * @param os
     * @param tree
     */
    template<typename TreeType>
    inline auto trace_group_dependencies(std::ostream& os, TreeType& tree) -> void
    {
        std::cout << "Trace of the  group dependencies\n";

        init_trace_group_dependencies(tree);

        os << "========================== M2L Group dependencies ========================= \n";

        const std::size_t top_level = tree.box().is_periodic() ? 0 : 2;
        const auto leaf_level = tree.leaf_level();
        auto& key_mul = tree.m_multipoles_dependencies;
        auto& key_loc = tree.m_locals_dependencies;
        for(std::size_t level = leaf_level; level >= top_level; --level)
        {
            os << "========== level : " << level << " ============================\n";
            const auto group_of_cell_begin = tree.cbegin_mine_cells(level);
            const auto group_of_cell_end = tree.cend_mine_cells(level);
            std::for_each(
              group_of_cell_begin, group_of_cell_end,
              [&os, level, leaf_level, top_level, &key_mul, &key_loc](auto const& ptr_group)
              {
                  auto const& current_group_symbolics = ptr_group->csymbolics();
                  os << "*** Group of cell index " << current_group_symbolics.idx_global_tree << " *** index in ["
                     << current_group_symbolics.starting_index << ", " << current_group_symbolics.ending_index << "[";
                  os << ", is_mine: " << std::boolalpha << current_group_symbolics.is_mine << "\n";
                  //   os << "    group size:  " << current_group_symbolics.number_of_component_in_group << ", ";
                  //   os << "global index (level)=  " << current_group_symbolics.idx_global << " \n";

                  //   std::ostringstream address_mult, address_loc;
                  //   address_mult << static_cast<void const*>(&(ptr_group->ccomponent(0).cmultipoles(0)));
                  //   address_loc << static_cast<void const*>(&(ptr_group->ccomponent(0).clocals(0)));

                  //   std::cout << " depend on multipoles: " << &(ptr_group->ccomponent(0).cmultipoles(0)) << " grp "
                  //             << key_mul[address_mult.str()] << " \n"
                  //             << " depend on locals:      " << &(ptr_group->ccomponent(0).clocals(0)) << " grp "
                  //             << key_loc[address_loc.str()] << " \n";

                  print_with_map(os, " m2l depend(in) ", current_group_symbolics.group_dependencies_m2l, key_mul);

                  if(level != leaf_level)
                  {
                      print_with_map(os, " m2m depend(in) ", current_group_symbolics.group_dependencies_m2m_in,
                                     key_mul);
                  }
                  if(level != top_level)
                  {
                      print_with_map(os, " l2l depend(in) ", current_group_symbolics.group_dependencies_l2l_in,
                                     key_loc);
                  }
                  os << std::endl;
              });
        }
    }
}   // namespace scalfmm::io
