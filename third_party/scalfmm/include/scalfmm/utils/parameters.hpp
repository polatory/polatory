// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/parameters.hpp
// --------------------------------
#ifndef EXAMPLES_PARAMETERS_HPP
#define EXAMPLES_PARAMETERS_HPP

#include <cpp_tools/cl_parser/cl_parser.hpp>

namespace args
{
    /**
     * @brief
     *
     */
    struct tree_height : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--tree-height", "-th"};
        std::string description = "Tree height (or initial height in case of an adaptive tree).";
        using type = int;
        type def = 2;
    };

    /**
     * @brief
     *
     */
    struct order : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--order", "-o"};
        std::string description = "Precision setting.";
        using type = std::size_t;
        type def = 3;
    };

    /**
     * @brief
     *
     */
    struct thread_count
    {
        cpp_tools::cl_parser::str_vec flags = {"--threads", "-t"};
        std::string description = "Maximum thread count to be used.";
        using type = std::size_t;
        type def = 1;
    };

    /**
     * @brief
     *
     */
    struct input_file : cpp_tools::cl_parser::required_tag
    {
        cpp_tools::cl_parser::str_vec flags = {"--input-file", "-fin"};
        std::string description = "Input filename (.fma or .bfma).";
        using type = std::string;
    };

    /**
     * @brief
     *
     */
    struct output_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--output-file", "-fout"};
        std::string description = "Output particle file (with extension .fma (ascii) or bfma (binary).";
        using type = std::string;
        type def = "output.fma";
    };

    /**
     * @brief
     *
     */
    struct log_level
    {
        cpp_tools::cl_parser::str_vec flags = {"--log-level", "-llog"};
        std::string description = "Log level to print.";
        using type = std::string;
        type def = "info";
    };

    /**
     * @brief
     *
     */
    struct log_file
    {
        cpp_tools::cl_parser::str_vec flags = {"--log-file", "-flog"};
        std::string description = "Log to file using spdlog.";
        using type = std::string;
        type def = "";
    };

    /**
     * @brief
     *
     */
    struct block_size
    {
        cpp_tools::cl_parser::str_vec flags = {"--group-size", "--block-size", "-gs", "-bs"};
        std::string description = "Group tree chunk size.";
        using type = int;
        type def = 250;
    };

    /**
     * @brief
     *
     */
    struct pbc
    {
        cpp_tools::cl_parser::str_vec flags = {"--per", "--pbc"};                         /*!< The flags */
        std::string description = "The periodicity in each direction (0 no periodicity)"; /*!< The description
                                                                                           */
        std::string input_hint = "0,1,1";                                                 /*!< The description */
        using type = std::vector<bool>;
    };

    /**
     * @brief
     *
     */
    struct extended_tree_height
    {
        using type = int;
        cpp_tools::cl_parser::str_vec flags = {"--ext-tree-height", "-eth"}; /*!< The flags */
        std::string description =
          "The number of level above the root level (used for periodicity)"; /*!< The description
                                                                              */
        std::string input_hint = "3";
        type def = 0;   // defaut no level for non periodic simulation
    };

    /**
     * @brief
     *
     */
    struct Dimension
    {
        cpp_tools::cl_parser::str_vec flags = {"--dimension", "-d"};
        std::string description = "Dimension : \n   2 for dimension 2, 3 for dimension 3";
        using type = int;
        type def = 3;
    };

}   // namespace args

#endif   // EXAMPLES_PARAMETERS_HPP
