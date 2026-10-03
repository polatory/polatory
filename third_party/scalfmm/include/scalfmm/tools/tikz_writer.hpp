// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tools/tikz_writer.hpp
// --------------------------------
#ifndef SCALFMM_TOOLS_TIKZ_WRITER_HPP
#define SCALFMM_TOOLS_TIKZ_WRITER_HPP

#include "scalfmm/container/access.hpp"
#include "scalfmm/meta/utils.hpp"
#include "scalfmm/tree/box.hpp"
#include "scalfmm/tree/for_each.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace scalfmm::tools::io
{
    /**
     * @brief export in TIKZ the leaves and their morton index.
     *
     * Generate a tikz pictures od the non empty leaves and the
     *  grid full parent grid.
     *
     * @tparam TreeType
     * @param filename name to store the tikz command.
     * @param tree to draw the leaf level.
     * @param displayParticles
     * @param color
     * @param plot_parent
     */
    template<class TreeType>
    auto exportTIKZ(const std::string& filename, const TreeType& tree, bool const displayParticles,
                    std::string const& color, const bool plot_parent = false) -> void
    {
        std::cout << "Write tikz in " << filename << std::endl;
        std::ofstream out(filename);

        auto width = tree.box_width() * 0.5;
        auto scale = 0.5 / tree.leaf_width(0);
        width *= scale;
        auto center = scale * tree.box_center();
        auto corner_l = center - width;
        auto corner_u = center + width;
        auto shift(corner_l);
        std::cout << "center " << center << std::endl;
        std::cout << "corner_l " << corner_l << "  corner_u " << corner_u << std::endl;

        shift *= -1.0;
        std::cout << "shift " << shift << " leaf size " << scale * tree.leaf_width() << " scale " << scale << std::endl;
        out << "\\begin{tikzpicture}[help lines/.style={blue!50,very thin}] " << std::endl;
        auto half_width = scale * tree.leaf_width() / 2.0;
        // std::cout << "shift " << shift << "leaf size " << scale * tree.leaf_width() << " half_width " << half_width
        //           << std::endl;

        component::for_each(
          std::get<0>(tree.begin()), std::get<0>(tree.end()),
          [&scale, &half_width, &out, &shift, &color, &displayParticles](auto& group)
          {
              //  std::size_t index_in_group{0};
              component::for_each(
                std::begin(*group), std::end(*group),
                [&scale, &half_width, &out, &shift, &color, &displayParticles](auto& leaf)
                {
                    auto center = scale * leaf.center();
                    auto corner_l = center - half_width;
                    auto corner_u = center + half_width;
                    out << "\\filldraw[fill=black!30!white]  (" << shift[0] + corner_l[0] << ","
                        << shift[1] + corner_l[1] << ")   rectangle (" << shift[0] + corner_u[0] << ","
                        << shift[1] + corner_u[1] << " );" << std::endl;

                    out << "\\node[scale=0.8,color=" << color << "] at (" << shift[0] + center[0] << ","
                        << shift[1] + center[1] << ")  {\\textbf{" << leaf.index() << "}};" << std::endl;
                    //
                    // Plot particles
                    if(displayParticles)
                    {
                        std::cout << " leaf " << leaf.index() << " nb part: " << leaf.size() << " " << std::endl;
                        // #ifdef USE_VIEW
                        using proxy_type = typename TreeType::leaf_type::particle_type::proxy_type;
                        for(auto const& p: leaf)
                        {
                            auto pos = scale * proxy_type(p).position();
                            out << "\\node[scale=0.4,color=" << color << "] at (" << shift[0] + pos[0] << ","
                                << shift[1] + pos[1] << ")  {x};" << std::endl;
                        }
                        // #else
                        //                                     const auto container =  leaf.cparticles() ;

                        //                                     for (int i{0}; i < leaf.size() ; ++i)
                        //                                     {
                        //                                         auto pos =
                        //                                         container.particle(i).position(); out <<
                        //                                         "\\node[scale=0.4,color=" << color << "]
                        //                                         at (" << shift[0] + pos[0]
                        //                                             << "," << shift[1] + pos[1] << ")
                        //                                             {x};" << std::endl;
                        //                                     }
                        // #endif
                    }
                });
          });
        out << "\\draw[ thick, black, step=" << scale * tree.leaf_width() << "] (" << shift[0] + corner_l[0] << ","
            << shift[1] + corner_l[1] << ")   grid (" << shift[0] + corner_u[0] << "," << shift[1] + corner_u[1]
            << " );" << std::endl;
        if(plot_parent)
        {
            ///
            auto cell_level_it = std::get<1>(tree.begin()) + tree.leaf_level() - 1;
            auto group_of_cell_begin = std::begin(*cell_level_it);
            auto group_of_cell_end = std::end(*cell_level_it);
            half_width *= 2;
            component::for_each(group_of_cell_begin, group_of_cell_end,
                                [&out, &shift, &scale, &half_width](auto& group)
                                {
                                    component::for_each(std::begin(*group), std::end(*group),
                                                        [&out, &shift, &scale, &half_width](auto& cell)
                                                        {
                                                            auto center = scale * cell.center();
                                                            auto corner_l = center - half_width;
                                                            auto corner_u = center + half_width;
                                                            out << "\\draw[thick, blue]  (" << shift[0] + corner_l[0]
                                                                << "," << shift[1] + corner_l[1] << ")   rectangle ("
                                                                << shift[0] + corner_u[0] << ","
                                                                << shift[1] + corner_u[1] << " );" << std::endl;

                                                            out << "\\node[scale=1.1, blue] at ("
                                                                << shift[0] + center[0] << "," << shift[1] + center[1]
                                                                << ")  {\\textbf{" << cell.index() << "}};"
                                                                << std::endl;
                                                        });
                                });
        }
        std::cout << "corner_l " << corner_l << "  corner_u " << corner_u << std::endl;
        std::cout << "C1 " << shift + corner_l << "  C2 " << shift + corner_u << std::endl;

        out << "\\end{tikzpicture}" << std::endl;

        out.close();
    }

}   // namespace scalfmm::tools::io
#endif
