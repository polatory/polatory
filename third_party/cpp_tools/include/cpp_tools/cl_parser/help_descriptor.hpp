#ifndef CPP_TOOLS_CL_PARSER_HELP_DESCRIPTOR_HPP
#define CPP_TOOLS_CL_PARSER_HELP_DESCRIPTOR_HPP

#include <iostream>

#include "cpp_tools/cl_parser/tcli.hpp"

namespace cpp_tools::cl_parser
{
    namespace meta
    {
        /**
         * \brief Checks whether type exposes a `description` attribute
         *
         * \tparam T Type to inspect
         */
        template<class T>
        struct has_description
        {
            template<typename U>
            static constexpr auto check(U*) -> decltype(std::declval<U*>()->description, void(), true)
            {
                return true;
            }
            static constexpr bool check(...) { return false; }
            /// value is true if `T::description` exists
            enum
            {
                value = check((T*)0)
            };
        };

        /**
         * \brief Checks whether type exposes a `program_description` attribute
         *
         * \tparam T Type to inspect
         */
        template<class T>
        struct has_program_description
        {
            template<typename U>
            static constexpr auto check(U*) -> decltype(std::declval<U*>()->program_description, void(), true)
            {
                return true;
            }
            static constexpr bool check(...) { return false; }
            /// value is true if `T::program_description` exists
            enum
            {
                value = check((T*)0)
            };
        };

        /**
         * \brief Checks whether type exposes a `flags` attribute
         *
         * \tparam T Type to inspect
         */
        template<class T>
        struct has_flags
        {
            template<typename U>
            static constexpr auto check(U*) -> decltype(std::declval<U*>()->flags, void(), true)
            {
                return true;
            }
            static constexpr bool check(...) { return false; }
            /// value is true if `T::flags` exists
            enum
            {
                value = check((T*)0)
            };
        };

        /**
         * \brief Checks whether type exposes a `input_hint` attribute
         *
         * \tparam T Type to inspect
         */
        template<class T>
        struct has_input_hint
        {
            template<typename U>
            static constexpr auto check(U*) -> decltype(std::declval<U*>()->input_hint, void(), true)
            {
                return true;
            }
            static constexpr bool check(...) { return false; }
            /// value is true if `T::input_hint` exists
            enum
            {
                value = check((T*)0)
            };
        };

        /**
         * \brief Checks whether type exposes a `hidden` compile time constant
         *
         * \tparam T Type to inspect
         *
         * Example:
         *
         * ~~~{.cpp}
         * struct P {};
         * struct S {
         *    enum {hidden};
         * };
         *
         * is_hidden<P>::value; // false
         * is_hidden<S>::value; // true
         * ~~~
         */
        template<class T>
        struct is_hidden
        {
            template<typename U, int = U::hidden>
            static constexpr bool check(U*)
            {
                return true;
            }
            static constexpr bool check(...) { return false; }

            enum
            {
                value = check((T*)0)
            };
        };

    }   // namespace meta

    namespace detail
    {
        /**
         * \brief Parameters data tree
         *
         * A tree of the parameters data. Each node contains the description of
         * a parameter. Sub-parsers are internal nodes, other parameters are
         * leaves.
         */
        struct flag_tree_node
        {
            /// Parameter names and flags, sorted by length, with prefix prepended
            std::vector<std::string> flags;
            /// Parameter description
            std::string description = "";
            /// Parameter input hint text
            std::string input_hint = "";
            /// Sub-parser child node
            std::vector<flag_tree_node> children;

            bool optional = false;
            bool stackable = false;
            bool flagged = false;

            /**
             * \brief Set the node data from a parameter descriptor
             *
             * \tparam ParamDesc Parameter descriptor type
             *
             * \param param Parameter descriptor
             */
            template<class ParamDesc>
            void set(const ParamDesc& param)
            {
                this->optional = !meta::is_required<ParamDesc>::value;
                this->stackable = meta::is_stackable<ParamDesc>::value;
                this->flagged = meta::is_flagged<ParamDesc>::value;

                this->add_flags(param);

                this->set_description(param);
                this->set_input_hint(param);

                set_children(param);
            }

            /**
             * \brief Add flags to the flag list
             *
             * \param param Parameter descriptor
             */
            template<class ParamDesc, meta::use_if<meta::has_flags<ParamDesc>> = 0>
            void add_flags(const ParamDesc& param)
            {
                for(auto const & f: param.flags)
                {
                    this->flags.emplace_back(f);
                }
            }

            /**
             * \brief Add program name to the flags
             *
             * \param param Parameter descriptor
             */
            template<class ParamDesc, meta::not_use_if<meta::has_flags<ParamDesc>> = 0>
            void add_flags(const ParamDesc& param)
            {
                this->flags.emplace_back(param.program_name);
            }

            /**
             * \brief Set the parameter description when it exists
             *
             * \tparam ParamDesc Parameter descriptor type
             *
             * \param param Parameter descriptor
             */
            template<class ParamDesc, meta::use_if<meta::has_description<ParamDesc>> = 0>
            void set_description(const ParamDesc& param)
            {
                this->description = param.description;
            }

            /**
             * \brief Set the description for the top-level parser
             *
             * \tparam ParamDesc Parameter descriptor type
             *
             * \param param Parameter descriptor
             */
            template<class ParamDesc, meta::not_use_if<meta::has_description<ParamDesc>> = 0,
                     meta::use_if<meta::has_program_description<ParamDesc>> = 0>
            void set_description(const ParamDesc& param)
            {
                this->description = param.program_description;
            }

            /**
             * \brief No-op when the parameter description doesn't exist
             *
             * \tparam ParamDesc Parameter descriptor type
             *
             * \param param Parameter descriptor
             */
            template<class ParamDesc, meta::not_use_if<meta::has_description<ParamDesc>> = 0,
                     meta::not_use_if<meta::has_program_description<ParamDesc>> = 0>
            void set_description(const ParamDesc&)
            {
            }

            /**
             * \brief Set the parameter description when it exists
             *
             * \tparam ParamDesc Parameter descriptor type
             *
             * \param param Parameter descriptor
             */
            template<class ParamDesc, meta::use_if<meta::has_input_hint<ParamDesc>> = 0>
            void set_input_hint(const ParamDesc& param)
            {
                this->input_hint = param.input_hint;
            }

            /**
             * \brief No-op when the parameter input hint doesn't exist
             *
             * \tparam ParamDesc Parameter descriptor type
             *
             * \param param Parameter descriptor
             */
            template<class ParamDesc, meta::not_use_if<meta::has_input_hint<ParamDesc>> = 0>
            void set_input_hint(const ParamDesc&)
            {
                if(!this->flagged)
                {
                    this->input_hint = "value";
                }
            }

            /**
             * \brief Functor to call the set method from a parameter handle
             */
            struct set_child_from_handle
            {
                /**
                 * \brief Create a new child from node and set it from a handle
                 *
                 * Calls `child.set(handle.descriptor)`
                 */
                template<class Handle>
                void operator()(const Handle& handle, flag_tree_node& node)
                {
                    if(meta::is_hidden<decltype(handle.descriptor)>::value)
                    {
                        return;
                    }
                    node.children.emplace_back();
                    node.children.back().set(handle.descriptor);
                }
            };

            /**
             * \brief Create sub-parser sub-tree
             *
             * \tparam ParameterDescriptors Parameter pack
             *
             * \param parser sub-parser which children must be added to the tree
             */
            template<class Parser, meta::use_if<meta::is_parser<Parser>> = 0>
            void set_children(const Parser& parser)
            {
                utils::for_each_in_tuple(parser.handles, set_child_from_handle{}, *this);
            }

            /**
             * \brief Non parser overload, no-op
             *
             * \tparam T Parameter type
             *
             * \param unnamed unused
             */
            template<class T, meta::not_use_if<meta::is_parser<T>> = 0>
            void set_children(const T&)
            {
            }

            friend std::ostream& operator<<(std::ostream& os, const flag_tree_node& node)
            {
                static int indent_v = 0;
                std::string indent = '\n' + std::string(2 * indent_v, ' ');

                bool is_parser = node.children.size();
                os << indent << node.flags.at(0);
                if(node.description != "")
                {
                    os << indent << node.description;
                }

                if(is_parser)
                {
                    ++indent_v;
                }

                for(auto& i: node.children)
                {
                    os << i;
                }

                if(is_parser)
                {
                    --indent_v;
                }

                return os;
            };
        };
    }   // namespace detail

    /**
     * \brief Help parameter descriptor
     *
     * This descriptor inspects the parser and prints a help message describing the
     * program.
     *
     */
    struct help
    {
        /// Terminal formatting capacities
        enum format
        {
            reset,
            bold,
            underline,
            red,
            green,
            blue,
            default_color
        };

        /**
         * \brief Base class for text interface format
         *
         * Used when formatting must no be applied to the help message.
         */
        struct term_format
        {
            /**
             * \brief Gets the escape sequence for given format
             *
             * \return an empty string
             */
            virtual std::string get(format) { return ""; }
        };

        /**
         * \brief Coloured terminal text interface format
         *
         * Used when formatting can be applied to the help message.
         */
        struct colored_term : term_format
        {
            /**
             * \brief Gets the escape sequence for given format
             *
             * \return std::string containing the escape sequence.
             */
            virtual std::string get(format fmt) override
            {
                switch(fmt)
                {
                case reset:
                    return "\033[m";
                case bold:
                    return "\033[1m";
                case underline:
                    return "\033[4m";
                case red:
                    return "\033[31m";
                case green:
                    return "\033[32m";
                case blue:
                    return "\033[33m";
                case default_color:
                    return "\033[39m";
                default:
                    return "";
                }
            }
        };

        /// Unused type, mandatory per interface specification
        using type = bool;
        /// The parameter is a flag, it doesn't expect a following value
        enum
        {
            flagged
        };

        /// Flags associated with the parameter
        str_vec flags{"--help", "-h"};
        /// Parameter description
        std::string description = "Display this help message";

        /// Type erased parser tree
        detail::flag_tree_node flag_tree;
        /// Text format object
        term_format* fmt;

        /**
         * \brief Parameter visitor
         *
         * The help parameter has a special behaviour. It stops parsing and inspects
         * the parser to print the help message. At the end of this function, the
         * program exits.
         *
         * \param parser The current parser
         * \param args   The argument list as received by the program
         * \param current_arg Iterator to the current argument
         */
        template<class Parser, class ArgContainer, class Iterator>
        void visit(Parser& parser, ArgContainer& args, Iterator& current_arg)
        {
            (void)current_arg, (void)args;

            colored_term t{};   // TODO: use unique pointer
            this->fmt = &t;
            this->flag_tree.set(parser);
            this->flag_tree.optional = false;
            this->flag_tree.stackable = false;

            std::cout << fmt->get(bold) + "USAGE:\n" + fmt->get(reset);
            std::cout << "  " << short_description(flag_tree) << '\n' << '\n';
            std::cout << fmt->get(bold) + "DESCRIPTION:" + fmt->get(reset);
            std::cout << long_description(flag_tree, 0, 80) << '\n' << '\n';

            exit(-1);
        }

        /**
         * \brief The parser short description, help message first line.
         *
         * name required-arg [opt-arg] {stack-arg} {opt-sub-parser req-sub-arg [opt-sub-arg]}
         */
        std::string short_description(detail::flag_tree_node& node)
        {
            std::string desc = node.flags.at(0);

            if(node.input_hint.size() > 0)
            {
                desc += ' ' + node.input_hint;
            }

            for(auto& child: node.children)
            {
                desc += ' ' + short_description(child);
            }

            if(node.optional)
            {
                desc = '[' + desc + ']';
                if(node.stackable)
                {
                    desc += "...";
                }
            }
            else if(node.stackable)
            {
                desc = '{' + desc + "}...";
            }
            return desc;
        };

        /**
         * \brief Create a string containing all the flags of a parameter
         *
         * \param node Type erased node containing parameter information
         */
        std::string long_flag_string(const detail::flag_tree_node& node)
        {
            std::string str = fmt->get(bold) + node.flags[0] + fmt->get(reset);
            for(std::size_t i = 1; i < node.flags.size(); ++i)
            {
                str += std::string(", ") + fmt->get(bold) + node.flags[i] + fmt->get(reset);
            }
            str += ' ';
            str += fmt->get(underline) + node.input_hint;
            if(node.children.size() > 0)
            {
                str += std::string("sub-options...");
            }
            str += fmt->get(reset);

            return str;
        }

        /**
         * \brief Create parameter long description
         *
         * The text is formatted following the given indent and paragraph
         * width. Children parameters (in case of a sub parser) are recursively
         * included in the text with an increased indent.
         *
         * \param node   Type erase parameter descriptor
         * \param indent Level of indentation
         * \param width  Text width in character count
         *
         * \return A string containing the description of a parameter and its
         * sub-parameters.
         */
        std::string long_description(const detail::flag_tree_node& node, std::size_t indent, std::size_t width)
        {
            std::string whitespace = " \n\t\r";
            std::string flag_str = format(long_flag_string(node), indent, "    ", width);
            std::string desc = format(node.description, indent + 1, "    ", width);

            if(&node == &this->flag_tree)
            {
                flag_str = "";
            }

            // Remove whitespace at the end of the description
            auto it = desc.end() - 1;
            while(it != desc.begin() && whitespace.find(*it) != std::string::npos)
            {
                --it;
            }
            desc.erase(++it, desc.end());

            for(auto& child: node.children)
            {
                desc += '\n' + long_description(child, indent + 1, width);
            }

            return {flag_str + desc};
        };

        /**
         * \brief Format a string to fit given width and indentation.
         *
         * \param str    String to format
         * \param indent Indentation level
         * \param indent_unit Indentation unit
         * \param width  Line width, in characters
         *
         * The final indentation is `indent * indent_unit`.
         *
         * \return A new string containing the formated text.
         */
        std::string format(std::string str, std::size_t indent, const std::string& indent_unit, std::size_t width)
        {
            std::string whitespace = " \t\n\r";
            std::vector<std::string> lines;
            std::string indent_str{};
            for(std::size_t i = 0; i < indent; ++i)
            {
                indent_str += indent_unit;
            }

            const auto str_end = str.end();

            for(auto it = str.begin(); it != str.end();)
            {
                // while on whitespace, move right
                while(it != str_end && whitespace.find(*it) != std::string::npos)
                {
                    if(*it == '\n')
                    {
                        break;
                    }
                    ++it;
                }
                // save iterator position
                auto beg = it;
                // move width chars to the right
                while(it != str_end && static_cast<unsigned long>(it - beg) < (width - indent_str.size()))
                {
                    if(*it == '\n')
                    {
                        *it = ' ';
                        break;
                    }
                    else if(*it == '\r')
                    {
                        beg = it + 1;
                    }
                    ++it;
                }

                if(it != str_end)
                {
                    // while not on whitespace, move left
                    while(it != beg && whitespace.find(*it) == std::string::npos)
                    {
                        --it;
                    }
                    bool moved_left = false;
                    // move left until previous non whitespace character
                    while(it != beg && whitespace.find(*it) != std::string::npos)
                    {
                        moved_left = true;
                        --it;
                    }
                    if(moved_left)
                    {
                        ++it;
                    }
                }

                lines.emplace_back("");
                lines.back().append(beg, it);
            }

            std::string res;
            for(auto& l: lines)
            {
                res += '\n' + indent_str + l;
            }
            return res;
        }
    };

}   // namespace cpptools::cl_parser

#endif /* TCLI_HELP_DESCRIPTOR_HPP */
