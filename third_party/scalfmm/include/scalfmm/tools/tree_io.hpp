// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tools/tree_io.hpp
// --------------------------------
#pragma once

#include "scalfmm/tree/utils.hpp"
#include "scalfmm/utils/io_helpers.hpp"

#include "scalfmm/tree/tree.hpp"
#ifdef SCALFMM_USE_MPI
#include <cpp_tools/parallel_manager/parallel_manager.hpp>
#endif

#include <array>
#include <fstream>
#include <functional>
#include <string>

namespace scalfmm::tools::io
{
    /**
     * @brief write the header of the tree in binary file.
     *
     * @tparam TreeType
     * @param out[inout]   the writing stream
     * @param tree[in]  the tree to write
     * @param header[in] the comments to write
     * @return auto
     */
    template<typename TreeType>
    inline auto write_binary_header(std::fstream& out, TreeType const& tree, std::string const& header) -> void
    {
        static constexpr std::size_t dimension = TreeType::dimension;
        static constexpr std::size_t inputs_size = TreeType::cell_type::storage_type::inputs_size;
        static constexpr std::size_t outputs_size = TreeType::cell_type::storage_type::outputs_size;
        using value_type = typename TreeType::position_value_type;
        // std::cout << " inputs_size1 " << inputs_size << " outputs_size " << outputs_size << std::endl;
        std::int32_t l = header.size();
        out.write(reinterpret_cast<char*>(&l), sizeof(std::int32_t));
        out.write(reinterpret_cast<const char*>(header.c_str()), l * sizeof(char));

        std::int32_t h = tree.height();
        out.write(reinterpret_cast<char*>(&h), sizeof(std::int32_t));
        h = tree.order();
        out.write(reinterpret_cast<char*>(&h), sizeof(std::int32_t));
        h = tree.group_of_leaf_size();
        out.write(reinterpret_cast<char*>(&h), sizeof(std::int32_t));
        h = tree.group_of_cell_size();
        out.write(reinterpret_cast<char*>(&h), sizeof(std::int32_t));
        out.write(reinterpret_cast<const char*>(&inputs_size), sizeof(std::int32_t));
        out.write(reinterpret_cast<const char*>(&outputs_size), sizeof(std::int32_t));
        //
        auto box = tree.box();
        auto c1 = box.c1();
        auto c2 = box.c2();
        out.write(reinterpret_cast<char*>(&c1[0]), dimension * sizeof(value_type));
        out.write(reinterpret_cast<char*>(&c2[0]), dimension * sizeof(value_type));
    }

    /**
     * @brief
     *
     * @tparam TreeType
     * @param out
     * @param tree
     * @param header
     * @return auto
     */
    template<typename TreeType>
    inline auto write_txt_header(std::fstream& out, TreeType const& tree, std::string const& header) -> void
    {
        static constexpr std::size_t dimension = TreeType::dimension;
        static constexpr std::size_t inputs_size = TreeType::cell_type::storage_type::inputs_size;
        static constexpr std::size_t outputs_size = TreeType::cell_type::storage_type::outputs_size;
        using value_type = typename TreeType::position_value_type;
        // std::cout << " inputs_size1 " << inputs_size << " outputs_size " << outputs_size << std::endl;
        out << header << std::endl;
        out << "h: " << tree.height() << std::endl;
        out << "o: " << tree.order() << std::endl;
        out << "gs :" << tree.group_of_leaf_size() << " " << tree.group_of_cell_size() << std::endl;
        out << "input_size: " << inputs_size << " output_size: " << outputs_size << std::endl;
        //
        out << tree.box().c1() << std::endl;
        out << tree.box().c2() << std::endl;
    }

    /**
     * @brief read the header of the tree and empty tree (without the hierarchy and the components)
     *
     * @tparam TreeType
     * @param[in] input the reading stream
     * @return TreeType
     */
    template<typename TreeType>
    inline auto read_binary_header(std::fstream& input) -> TreeType
    {
        static constexpr int dimension = TreeType::dimension;
        static constexpr std::size_t inputs_size = TreeType::cell_type::storage_type::inputs_size;
        static constexpr std::size_t outputs_size = TreeType::cell_type::storage_type::outputs_size;
        using value_type = typename TreeType::position_value_type;
        using position_type = typename TreeType::position_type;
        using box_type = typename TreeType::box_type;
        std::cout << "dimension : " << dimension << std::endl;
        std::int32_t l;
        input.read(reinterpret_cast<char*>(&l), sizeof(std::int32_t));
        char* t = new char[l];
        input.read(t, l * sizeof(char));
        std::string header(t);
        std::clog << "Header" << header << '\n';
        //
        constexpr int size_to_read{6};
        // height, order, gs_leaf, gs_cell, inputs_size, outputs_size
        std::array<std::int32_t, size_to_read> in;
        // height, top_level
        input.read(reinterpret_cast<char*>(in.data()), size_to_read * sizeof(std::int32_t));
        // scalfmm::io::print(std::clog, in);
        // check
        std::cout << "in: ";
        for(auto const e: in)
        {
            std::cout << " " << e;
        }
        std::cout << std::endl;
        if(inputs_size != in[4] or outputs_size != in[5])
        {
            std::cerr << "\n Error wrong cell type !\n";
            std::cerr << " Cell::inputs_size  " << inputs_size << "  read " << in[4] << '\n'
                      << " Cell::outputs_size " << outputs_size << "  read " << in[5] << std::endl;
            std::exit(EXIT_FAILURE);
        }
        //
        std::array<value_type, 2 * dimension> w;
        input.read(reinterpret_cast<char*>(w.data()), 2 * dimension * sizeof(value_type));
        // scalfmm::io::print(std::clog, w);
        position_type c1, c2;
        for(int d = 0; d < dimension; ++d)
        {
            c1[d] = w[d];
            c2[d] = w[dimension + d];
        }
        box_type box(c1, c2);
        std::cout << "box: " << box << std::endl;
        return TreeType(in[0], in[1], in[2], in[3], box);
    }
    /**
     * @brief save a tree in binary format in a file (only the cells).
     *
     * @tparam CellType
     * @tparam LeafType
     * @tparam BoxType
     * @param filename[in]   name of the file
     * @param tree[in]   tree to save
     * @param header[in]   string that represents a comment (useful to know what tree we read)
     * @param options[in]   (not use yet)
     */
    template<typename CellType, typename LeafType, typename BoxType>
    inline auto save_bin(std::string const& filename,
                         scalfmm::component::group_tree_view<CellType, LeafType, BoxType> const& tree,
                         std::string const& header, const int options = 0) -> void
    {
        std::cout << "save tree (binary mode) in filename " << filename << std::endl;
        std::fstream out(filename, std::ifstream::out | std::ios::binary);
        write_binary_header(out, tree, header);
        // Compute and save the number of cells per level
        std::vector<std::int64_t> number_of_cells(tree.height(), int(0));
        auto group_size = tree.group_of_cell_size();
        for(std::size_t level = tree.top_level(); level < tree.height(); ++level)
        {
            auto begin = tree.begin_mine_cells(level);
            auto end = tree.end_mine_cells(level) - 1;
            number_of_cells[level] = std::distance(begin, end) * group_size;
            // The last group may not be complete
            number_of_cells[level] += end->get()->size();
        }
        out.write(reinterpret_cast<char*>(number_of_cells.data()), number_of_cells.size() * sizeof(std::int64_t));
        //
        using morton_type = std::int64_t;
        // Save the morton index at the leaf level
        {
            auto level = tree.leaf_level();
            std::vector<morton_type> morton(number_of_cells[level]);
            // Save the morton indexes
            for(auto grp = tree.begin_mine_cells(level); grp != tree.end_mine_cells(level); ++grp)
            {
                for(std::size_t index = 0; index < (*grp)->size(); ++index)
                {
                    auto const& cell = (*grp)->ccomponent(index);
                    morton[index] = cell.csymbolics().morton_index;
                }
                out.write(reinterpret_cast<char*>(morton.data()), (*grp)->size() * sizeof(morton_type));
            }
        }
        // loop on the groups
        using value_type = typename CellType::value_type;
        for(std::size_t level = tree.leaf_level(); level >= tree.top_level(); --level)
        {
            // std::cout << "level " << level << std::endl;
            for(auto grp = tree.begin_mine_cells(level); grp != tree.end_mine_cells(level); ++grp)
            {
                for(std::size_t index = 0; index < (*grp)->size(); ++index)
                {
                    auto const& cell = (*grp)->ccomponent(index);
                    auto mults = cell.cmultipoles();
                    auto number_of_multipole = mults.size();
                    for(std::size_t l = 0; l < number_of_multipole; ++l)
                    {
                        auto& mult = mults.at(l);
                        auto shape = mult.shape();
                        auto nb = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                        out.write(reinterpret_cast<char*>(mult.data()), nb * sizeof(value_type));
                    }
                    auto locals = cell.clocals();
                    number_of_multipole = locals.size();

                    for(std::size_t l = 0; l < number_of_multipole; ++l)
                    {
                        auto& local = locals.at(l);
                        auto shape = local.shape();
                        auto nb = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                        out.write(reinterpret_cast<char*>(local.data()), nb * sizeof(value_type));
                    }
                }
            }
        }
        out.close();
    }

    /**
     * @brief save a tree in text format in a file (only the cells).
     *
     * @tparam CellType
     * @tparam LeafType
     * @tparam BoxType
     * @param filename[in]   name of the file
     * @param tree[in]   tree to save
     * @param header[in]   string that represents a comment (useful to know what tree we read)
     * @param options[in]   (not use yet)
     */
    template<typename CellType, typename LeafType, typename BoxType>
    inline auto save_txt(std::string const& filename,
                         component::group_tree_view<CellType, LeafType, BoxType> const& tree, std::string const& header,
                         const int options = 0) -> void
    {
        std::cout << "save tree (ascii mode) in filename " << filename << std::endl;

        std::fstream out(filename, std::ifstream::out | std::ios::binary);
        write_txt_header(out, tree, header);
        // Compute and save the number of cells per level
        std::vector<std::int64_t> number_of_cells(tree.height(), int(0));
        auto group_size = tree.group_of_cell_size();
        for(std::size_t level = tree.top_level(); level < tree.height(); ++level)
        {
            auto begin = tree.begin_mine_cells(level);
            auto end = tree.end_mine_cells(level) - 1;
            number_of_cells[level] = std::distance(begin, end) * group_size;
            // The last group may not be complete
            number_of_cells[level] += end->get()->size();
        }
        scalfmm::io::print(out, "nb cells per level ", number_of_cells.begin(), number_of_cells.end());
        out << std::endl;
        //
        using morton_type = std::int64_t;
        // Save the morton index at the leaf level
        {
            auto level = tree.leaf_level();
            std::vector<morton_type> morton(number_of_cells[level]);
            // Save the morton indexes
            for(auto grp = tree.begin_mine_cells(level); grp != tree.end_mine_cells(level); ++grp)
            {
                for(std::size_t index = 0; index < (*grp)->size(); ++index)
                {
                    auto const& cell = (*grp)->ccomponent(index);
                    morton[index] = cell.csymbolics().morton_index;
                }
            }
            scalfmm::io::print(out, "morton(leaf) ", morton.begin(), morton.end(), ", ");
            out << std::endl;
        }
        // loop on the groups
        using value_type = typename CellType::value_type;
        for(std::size_t level = tree.leaf_level(); level >= tree.top_level(); --level)
        {
            // std::cout << "level " << level << std::endl;
            for(auto grp = tree.begin_mine_cells(level); grp != tree.end_mine_cells(level); ++grp)
            {
                for(std::size_t index = 0; index < (*grp)->size(); ++index)
                {
                    auto const& cell = (*grp)->ccomponent(index);
                    auto mults = cell.cmultipoles();
                    auto number_of_multipole = mults.size();

                    for(std::size_t l = 0; l < number_of_multipole; ++l)
                    {
                        auto mult = mults.at(l);
                        auto shape = mult.shape();
                        auto nb = shape[0] * shape[1];
                        out << mult << std::endl;
                    }
                    auto locals = cell.clocals();

                    for(std::size_t l = 0; l < number_of_multipole; ++l)
                    {
                        auto local = locals.at(l);
                        auto shape = local.shape();
                        auto nb = shape[0] * shape[1];
                        out << local << std::endl;
                    }
                }
            }
        }
        out.close();
    }

    /**
     * @brief save a tree in either text or binary format in a file (only the cells).
     *
     * @tparam CellType
     * @tparam LeafType
     * @tparam BoxType
     * @param filename[in]   name of the file with extention .bib or .txt
     * @param tree[in]   tree to save
     * @param header[in]   string that represents a comment (useful to know what tree we read)
     * @param options[in]   (not use yet)
     */
    template<typename CellType, typename LeafType, typename BoxType>
    inline auto save(std::string const& filename, component::group_tree_view<CellType, LeafType, BoxType> const& tree,
                     std::string const& header, const int options = 0) -> void
    {
        if(filename.find(".bin") != std::string::npos)
        {
            save_bin(filename, tree, header, options);
        }
        else if(filename.find(".txt") != std::string::npos)
        {
            save_txt(filename, tree, header, options);
        }
        else
        {
            std::cout << "scalfmm::tools::io::save:\n "
                      << "Only .bin or .txt  file are allowed. Got " << filename << "." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    /**
     * @brief Read a tree (only the cells) and set a tree from a file
     *
     * @tparam TreeType
     * @param[in]  filename name of the file containing the ree
     * @param[in]  options (unused)
     * @return TreeType
     */
    template<typename TreeType>
    inline auto read(std::string const& filename, const int options = 0) -> TreeType
    {
        using morton_type = std::size_t;   // int64_t;
        std::cout << " Read from file " << filename << std::endl;
        std::fstream input(filename, std::ifstream::in | std::ios::binary);

        auto tree(read_binary_header<TreeType>(input));

        std::vector<std::int64_t> number_of_cells(tree.height(), std::int64_t(0));
        input.read(reinterpret_cast<char*>(number_of_cells.data()), number_of_cells.size() * sizeof(std::int64_t));
        std::vector<morton_type> vector_of_mortons(number_of_cells[tree.height() - 1]);
        input.read(reinterpret_cast<char*>(vector_of_mortons.data()), vector_of_mortons.size() * sizeof(morton_type));
        // Construct the structure of blocs at all levels inside the tree
        scalfmm::io::print("vector_of_mortons", vector_of_mortons);

        tree.construct(vector_of_mortons);
        //
        // Fill tree cells
        using value_type = typename TreeType::cell_type::value_type;

        for(std::size_t level = tree.leaf_level(); level >= tree.top_level(); --level)
        {
            // std::cout << "level " << level << std::endl;
            for(auto grp = tree.begin_mine_cells(level); grp != tree.end_mine_cells(level); ++grp)
            {
                for(std::size_t index = 0; index < (*grp)->size(); ++index)
                {
                    auto& cell = (*grp)->component(index);

                    auto& mults = cell.multipoles();
                    auto number_of_arrays = mults.size();
                    // std::cout << "number_of_arrays " << number_of_arrays << std::endl;
                    for(std::size_t l = 0; l < number_of_arrays; ++l)
                    {
                        auto& mult = mults.at(l);
                        auto nb = mult.size();
                        input.read(reinterpret_cast<char*>(mult.data()), nb * sizeof(value_type));
                        // std::cout << "mult\n" << mult << std::endl;
                    }
                    auto& locals = cell.locals();
                    number_of_arrays = locals.size();
                    // std::cout << "locals number_of_arrays " << number_of_arrays << std::endl;

                    for(std::size_t l = 0; l < number_of_arrays; ++l)
                    {
                        auto& local = locals.at(l);
                        auto nb = local.size();
                        input.read(reinterpret_cast<char*>(local.data()), nb * sizeof(value_type));
                        // std::cout << "local \n" << local << std::endl;
                    }
                }
            }
        }
        input.close();
        return tree;
    }

#ifdef SCALFMM_USE_MPI

    /**
     * @brief save a tree in binary format in a file (only the cells).
     *
     * @tparam CellType
     * @tparam LeafType
     * @tparam BoxType
     * @param[in]  para the parallel manager
     * @param[in]  filename  name of teh file
     * @param[out] tree  tree to save
     * @param[in]  header  string that represents a comment (useful to know what tree we read)
     * @param[in]  options  (not use yet)
     */
    template<typename CellType, typename LeafType, typename BoxType>
    inline auto save(cpp_tools::parallel_manager::parallel_manager& para, std::string const& filename,
                     component::dist_group_tree<CellType, LeafType, BoxType> const& tree, std::string const& header,
                     const int options = 0) -> void
    {
        static constexpr std::size_t inputs_size =
          component::dist_group_tree<CellType, LeafType, BoxType>::cell_type::storage_type::inputs_size;
        static constexpr std::size_t outputs_size =
          component::dist_group_tree<CellType, LeafType, BoxType>::cell_type::storage_type::outputs_size;
        if(para.master())
        {
            std::fstream out(filename, std::ifstream::out | std::ios::binary);
            // std::fstream out(filename, std::ifstream::out);
            out.close();
        }
        auto comm = para.get_communicator();

        comm.barrier();
        // The file exists, we can read it
        std::fstream out(filename, std::ifstream::out | std::ios::binary);
        // std::fstream out(filename, std::ifstream::out);

        std::int64_t pos = -1;
        if(para.master())
        {
            write_binary_header(out, tree, header);
        }
        // Comm
        MPI_Datatype int64_datatype = cpp_tools::parallel_manager::mpi::get_datatype<std::int64_t>();

        // Compute and save the number of cells per level
        std::vector<std::int64_t> number_of_cells(tree.height(), std::int64_t(0));
        auto group_size = tree.group_of_cell_size();
        for(int level = tree.top_level(); level < tree.height(); ++level)
        {
            auto begin = tree.begin_mine_cells(level);
            auto end = tree.end_mine_cells(level) - 1;
            number_of_cells[level] = std::distance(begin, end) * group_size;
            // The last group may not be complete
            number_of_cells[level] += end->get()->size();
        }
        // scalfmm::io::print("number_of_cells: ", number_of_cells);
        std::vector<std::int64_t> glob_number_of_cells(tree.height(), std::int64_t(0));

        comm.allreduce(number_of_cells.data(), glob_number_of_cells.data(), tree.height(), int64_datatype, MPI_SUM);
        if(para.master())
        {
            out.write(reinterpret_cast<char*>(glob_number_of_cells.data()),
                      number_of_cells.size() * sizeof(std::int64_t));
        }
        // scalfmm::io::print("glob_number_of_cells: ", glob_number_of_cells);
        std::vector<std::int64_t> scan_number_of_cells(tree.height(), std::int64_t(0));

        comm.scan(number_of_cells.data(), scan_number_of_cells.data(), tree.height(), int64_datatype, MPI_SUM);
        // scalfmm::io::print(" scan_number_of_cells" + std::to_string(rank) + ": ", scan_number_of_cells);
        for(int level = tree.top_level(); level < tree.height(); ++level)
        {
            scan_number_of_cells[level] -= number_of_cells[level];
        }
        //
        using morton_type = std::int64_t;
        //
        if(para.master())
        {
            pos = out.tellp();
        }
        comm.bcast(&pos, 1, int64_datatype, 0);
        // std::cout << "pos " << pos << std::endl;
        //
        auto leaf_level = tree.leaf_level();
        auto pos_start = pos;

        // Save the morton index at the leaf level
        {
            pos += scan_number_of_cells[leaf_level] * sizeof(morton_type);
            // std::cout << "pos " << pos << std::endl;
            out.seekp(pos);
            // std::cout << "out.seekp(pos) " << std::endl;

            std::vector<morton_type> morton(tree.group_of_cell_size());
            // Save the morton indexes
            for(auto grp = tree.begin_mine_cells(leaf_level); grp != tree.end_mine_cells(leaf_level); ++grp)
            {
                for(std::size_t index = 0; index < (*grp)->size(); ++index)
                {
                    auto const& cell = (*grp)->ccomponent(index);
                    morton[index] = cell.csymbolics().morton_index;
                }
                out.write(reinterpret_cast<char*>(morton.data()), (*grp)->size() * sizeof(morton_type));
            }
        }
        pos_start += sizeof(morton_type) * glob_number_of_cells[leaf_level];
        {
            using value_type = typename CellType::value_type;

            // loop on the groups
            // nb = size of the multipoles and the locals in a cell
            auto nb = tree.order() * tree.order() * (inputs_size + outputs_size);

            for(int level = tree.leaf_level(); level >= tree.top_level(); --level)
            {
                out.seekp(pos_start);
                // std::cout << rank << " true cur pos  " << pos << " pos_start  " << pos_start << std::endl;
                pos = pos_start + scan_number_of_cells[level] * nb * sizeof(value_type);

                out.seekp(pos);
                // std::cout << rank << " pos final(level=" << level << ") " << pos << std::endl;
                // std::cout << "level " << level << std::endl;
                for(auto grp = tree.begin_mine_cells(level); grp != tree.end_mine_cells(level); ++grp)
                {
                    for(std::size_t index = 0; index < (*grp)->size(); ++index)
                    {
                        auto const& cell = (*grp)->ccomponent(index);
                        auto mults = cell.cmultipoles();
                        auto number_of_multipole = mults.size();

                        for(int l = 0; l < number_of_multipole; ++l)
                        {
                            auto mult = mults.at(l);
                            auto shape = mult.shape();
                            auto nb1 = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

                            out.write(reinterpret_cast<char*>(mult.data()), nb1 * sizeof(value_type));
                        }
                        auto locals = cell.clocals();
                        number_of_multipole = locals.size();

                        for(int l = 0; l < number_of_multipole; ++l)
                        {
                            auto local = locals.at(l);
                            auto shape = local.shape();
                            auto nb1 = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                            out.write(reinterpret_cast<char*>(local.data()), nb1 * sizeof(value_type));
                        }
                    }
                }
                pos_start += sizeof(value_type) * nb * glob_number_of_cells[level];
            }
        }
        pos = out.tellp();
        // std::cout << rank << " end of file= " << pos << std::endl;

        out.close();
    }

#endif

}   // namespace scalfmm::tools::io
