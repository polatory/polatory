#ifndef CPP_TOOLS_CL_PARSER_PARAMETER_MODEL_HPP
#define CPP_TOOLS_CL_PARSER_PARAMETER_MODEL_HPP

#include <string>
#include <vector>

namespace cpp_tools::cl_parser
{
    /**
     * \brief Parameter descriptor model
     *
     * A parameter descriptor is a class that exposes the following public
     * interface.
     */
    struct parameter_descriptor_model
    {
        /// argument type
        using type = int;

        /// flags
        std::vector<std::string> flags = {"t"};
        /// optional, argument description
        const char *description;
        /// optional, argument default value
        type def;

        enum
        {
            stackable, ///< optional, the argument may be specified several time
            ///< and the value are stacked
            required, ///< optional, the argument must be specified once
            hidden,   ///< optional, the argument must be hidden in help message
        };

        /// optional, called after parameter parsing
        template <class Parser>
        void visit(Parser &p, std::vector<std::string> &args, typename std::vector<std::string>::iterator &current_arg);

        /// optional, customize parameter parsing
        bool parse(std::vector<std::string> &args, typename std::vector<std::string>::iterator &current_arg,
                   type &data);
    };

    /**
     * \brief Basic parameter descriptor
     *
     * \tparam T Parameter underlying type.
     * \tparam Tag Optional tag class use to differentiate two parameters with
     *             the same type.
     */
    template <class T, class Tag = void>
    struct parameter_descriptor
    {
        using type = T;
        std::vector<std::string> flags;

        parameter_descriptor() = default;
        parameter_descriptor(const parameter_descriptor &) = default;
        parameter_descriptor(parameter_descriptor &&) = default;
        parameter_descriptor &operator=(const parameter_descriptor &) = default;
        parameter_descriptor &operator=(parameter_descriptor &&) = default;

        parameter_descriptor(const std::vector<std::string> &new_flags)
            : flags(new_flags)
        {
        }
    };

    /**
     * \brief Basic flag structure, inherit from it to quickly create a flag
     *
     * Structure meant to be inherited from to create a flag parameter. Flags do
     * not consume the following argument on the CLI.
     *
     * Creates a `bool` typed parameter which value defaults to `false`.
     */
    struct flag_descriptor
    {
        using type = bool;
        enum
        {
            flagged
        };
    };

} // namespace cpptools::cl_parser

#endif /* TCLI_PARAMETER_MODEL_HPP */
