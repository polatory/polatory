#ifndef CPP_TOOLS_CL_PARSER_TCLI_HPP
#define CPP_TOOLS_CL_PARSER_TCLI_HPP

#include <ostream>

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include "cpp_tools/cl_parser/exceptions.hpp"
#include "cpp_tools/cl_parser/impl_tcli/meta.hpp"
#include "cpp_tools/cl_parser/impl_tcli/utils.hpp"

namespace cpp_tools::cl_parser {
namespace detail {
// this is only require to handle vector of parameters.
template <typename ValueType>
std::istream &operator>>(std::istream &is, std::vector<ValueType> &t) {
  bool first = true;
  ValueType val;
  char delimiter;
  while (!is.eof()) {
    // The first object is not delimited
    first ? (first = false) : ((is >> delimiter), true);
    // Get the value
    is >> val;
    t.emplace_back(val);
  }
  return is;
}
inline std::istream &operator>>(std::istream &is, std::vector<std::string> &t) {
  char val{'\0'};
  std::string s;
  char delimiter{','};
  while (!is.eof()) {
    // Get the value
    is >> val;
    s.push_back(val);
    if (val == delimiter) {
      s.erase(s.size() - 1);
      t.emplace_back(s);
      s.clear();
    }
  }
  if (!s.empty()) {
    t.emplace_back(s);
  }
  return is;
}
/**
 * \brief Get container size
 */
template <class C>
constexpr auto size(const C &c) noexcept(noexcept(c.size()))
    -> decltype(c.size()) {
  return c.size();
}

/**
 * \brief Get array size
 */
template <class T, std::size_t N>
constexpr std::size_t size(const T (&)[N]) noexcept {
  return N;
}

/**
 * \brief Parameter data management
 *
 * This class helps managing a parameter's data. It is used at compile
 * time to extract the parameter type and exposes generic methods to
 * access data.
 */
template <class ParamDesc, bool = meta::is_stackable<ParamDesc>::value>
struct parameter_data;

/**
 * \brief Single parameter data management specialisation
 *
 * Single parameters' data is stored is an instance of the data type.
 *
 * \tparam ParamDesc The parameter descriptor type
 */
template <class ParamDesc> struct parameter_data<ParamDesc, false> {
  /// Parameter descriptor data type
  using type = typename ParamDesc::type;
  /// Data access
  static type &get(type &d) { return d; }
};

/**
 * \brief Stacked parameter data management specialisation
 *
 * Stacked parameters' data is stored in a vector. Each access creates a
 * new entry in the data vector.
 *
 * \tparam ParamDesc The parameter descriptor type
 */
template <class ParamDesc> struct parameter_data<ParamDesc, true> {
  using type = std::vector<typename ParamDesc::type>;
  /**
   * \brief Data access
   * \param Data container
   * \note A new data instance is created for each call
   */
  static typename ParamDesc::type &get(type &d) {
    d.emplace_back();
    return d.back();
  }
};

/**
 * \brief Shorthand for `typename parameter_data<T>::type`
 */
template <class ParamDesc>
using parameter_data_t = typename parameter_data<ParamDesc>::type;

// Forward declaration of parameter handle structure
template <class ParamDesc> struct parameter_handle;

/**
 * \brief Generic single parse function
 *
 * \note Not part of overload resolution if ParamDesc defines a parse
 * method or is marked as flagged.
 *
 * \tparam ParamDesc Parameter descriptor type to use for parsing
 * \tparam Data Parameter data type
 * \tparam Container Argument CLI string container
 * \tparam ForwardIt Iterator of Container
 *
 * \param handle Reference to the parameter handle
 * \param data Reference to the data to set
 * \param args String arguments vector
 * \param current_arg Current position in args
 */
template <template <class> class Handle, class ParamDesc, class Data,
          class Container, class ForwardIt,
          meta::not_use_if<meta::has_parse<
              ParamDesc, bool(Container &, ForwardIt &, Data &)>> = 0,
          meta::not_use_if<meta::is_flagged<ParamDesc>> = 0>
bool parse_parameter(Handle<ParamDesc> &handle, Data &data, Container &args,
                     ForwardIt &current_arg) {
  (void)args, (void)handle;
  std::stringstream sstr(*current_arg);
  sstr >> data;
  if (!sstr.eof()) {
    std::stringstream msg;
    msg << "Partial parse for " << *(current_arg - 1) << ": ";
    msg << "argument given is '" << sstr.str() << "'";

    if (sstr.tellg() != typename std::stringstream::pos_type(-1)) {
      msg << ", remaining part is '" << sstr.str().substr(sstr.tellg()) << "'";
    }

    throw partial_parse(msg.str());
  }
  ++current_arg;
  return true;
}
// Bad specialization in the case of std::string to parse the full string...
// TODO: find a better approach
template <template <class> class Handle, class ParamDesc, class Container,
          class ForwardIt,
          meta::not_use_if<meta::has_parse<
              ParamDesc, bool(Container &, ForwardIt &, std::string &)>> = 0,
          meta::not_use_if<meta::is_flagged<ParamDesc>> = 0>
bool parse_parameter(Handle<ParamDesc> &handle, std::string &data,
                     Container &args, ForwardIt &current_arg) {
  (void)args, (void)handle;
  data = *(current_arg);
  ++current_arg;
  return true;
}

/**
 * \brief Single flagged parameter parse function
 *
 * \note Not part of overload resolution if ParamDesc defines a parse
 * method.
 *
 * \tparam ParamDesc Parameter descriptor type to use for parsing
 * \tparam Data Parameter data type
 * \tparam Container Argument CLI string container
 * \tparam ForwardIt Iterator of Container
 *
 * \param handle Reference to the parameter handle
 * \param data Reference to the data to set
 * \param args String arguments vector
 * \param current_arg Current position in args
 */
template <template <class> class Handle, class ParamDesc, class Data,
          class Container, class ForwardIt,
          meta::not_use_if<meta::has_parse<
              ParamDesc, bool(Container &, ForwardIt &, Data &)>> = 0,
          meta::use_if<meta::is_flagged<ParamDesc>> = 0>
bool parse_parameter(Handle<ParamDesc> &handle, Data &data, Container &args,
                     ForwardIt &current_arg) {
  (void)args, (void)handle, (void)current_arg;
  data = true;
  std::string flag = *(current_arg - 1);
  if (detail::size(flag) > 4) {
    if (flag.find(std::string("=off")) == detail::size(flag) - 4) {
      data = false;
    }
  }

  return true;
}

/**
 * \brief Single custom parameter parse function
 *
 * \note Not part of overload resolution if ParamDesc does not define a
 * parse method
 *
 * \tparam ParamDesc Parameter descriptor type to use for parsing
 * \tparam Data Parameter data type
 * \tparam Container Argument CLI string container
 * \tparam ForwardIt Iterator of Container
 *
 * \param handle Reference to the parameter handle
 * \param data Reference to the data to set
 * \param args String arguments vector
 * \param current_arg Current position in args
 */
template <template <class> class Handle, class ParamDesc, class Data,
          class Container, class ForwardIt,
          meta::use_if<meta::has_parse<ParamDesc, bool(Container &, ForwardIt &,
                                                       Data &)>> = 0>
bool parse_parameter(Handle<ParamDesc> &, Data &data, Container &args,
                     ForwardIt &current_arg) {
  bool success = ParamDesc::parse(args, current_arg, data);
  return success;
}

/**
 * \brief Call parameter visitor if it exists
 *
 * \tparam ParamDesc Parameter descriptor type to use for parsing
 * \tparam Parser Parser type
 * \tparam Container Argument CLI string container
 * \tparam ForwardIt Iterator of Container
 *
 * \param handle Reference to the parameter handle
 * \param args String arguments vector
 * \param current_arg Current position in args
 * \param p Parser from which the visitor is called
 *
 */
template <template <class> class Handle, class ParamDesc, class Parser,
          class Container, class ForwardIt,
          meta::use_if<meta::has_visit<ParamDesc, void(Parser &, Container &,
                                                       ForwardIt &)>> = 0>
void visit_parameter(Handle<ParamDesc> &handle, Container &args,
                     ForwardIt &current_arg, Parser &p) {
  handle.descriptor.visit(p, args, current_arg);
}

/**
 * \brief Call parameter visitor if it exists
 *
 * \tparam ParamDesc Parameter descriptor type to use for parsing
 * \tparam Parser Parser type
 * \tparam Container Argument CLI string container
 * \tparam ForwardIt Iterator of Container
 *
 * \param handle Reference to the parameter handle
 * \param args String arguments vector
 * \param current_arg Current position in args
 * \param p Parser from which the visitor is called
 *
 */
template <template <class> class Handle, class ParamDesc, class Parser,
          class Container, class ForwardIt,
          meta::not_use_if<meta::has_visit<
              ParamDesc, void(Parser &, Container &, ForwardIt &)>> = 0>
void visit_parameter(const Handle<ParamDesc> &handle, const Container &args,
                     const ForwardIt &current_arg, const Parser &p) {
  (void)handle, (void)args, (void)current_arg, (void)p;
}

/**
 * \brief Parameter data handle
 *
 * Used to move around a parameter data instanciation along with
 * additional information.
 *
 * \tparam ParamDesc Parameter descriptor
 */
template <class ParamDesc> struct parameter_handle {
  /// Data descriptor structure
  using data_descriptor = parameter_data<ParamDesc>;
  /// Parameter descritor
  using parameter_descriptor = ParamDesc;

  /// Parameter descriptor instance
  ParamDesc descriptor;
  /// Data for the parameter
  typename data_descriptor::type data;
  /// Parameter set flag, true if the parameter was seen once during parse.
  bool param_set = false;

  /**
   * \brief Parses an argument and assigns it to the data handled
   *
   * The argument data may be stored in a vector or in an single data
   * instance. SFINAE is used to distinguish the cases.
   *
   * \tparam ArgContainer Argument container type
   * \tparam ForwardIt Current argument iterator type
   *
   * \param args Container for the CLI arguments
   * \param current_arg Iterator to the current argument in args
   */
  template <class ArgContainer, class ForwardIt>
  void assign(ArgContainer &args, ForwardIt &current_arg) {
    auto &single_data = data_descriptor::get(data);
    try {
      this->param_set |= parse_parameter(*this, single_data, args, current_arg);
    } catch (unknown_parameter &) {
      // thrown when sub-parsers reach a parameter they don't know
      this->param_set = true;
    }
  }

  /**
   * \brief Calls a parameter visitor
   *
   * \tparam Parser Parser type
   * \tparam ArgContainer Argument container type
   * \tparam ForwardIt Current argument iterator type
   *
   * \param p Parser the argument was fed to
   * \param args Container for the CLI arguments
   * \param current_arg Iterator to the current argument in args
   */
  template <class Parser, class ArgContainer, class ForwardIt>
  void visit(ArgContainer &args, ForwardIt &current_arg, Parser &p) {
    visit_parameter(*this, args, current_arg, p);
  }

  friend std::ostream &operator<<(std::ostream &os, const parameter_handle &h) {
    os << '{';
    os << std::boolalpha << h.param_set;
    os << ',';
    os << h.data;
    os << '}';
    return os;
  }
};

/**
 * \brief Set parameter to its default value if it has one and has not
 * been set.
 */
struct set_default {
  std::vector<std::string> missing_required;

  /**
   * \brief If parameter has not been set, set it to it default value.
   *
   * \tparam Handle Parameter handle type
   *
   * \param handle Parameter handle
   *
   * \note This overload is called when a default value exists in the
   *       parameter descriptor: `handle.descriptor.def`.
   */
  template <class Handle, meta::use_if<meta::has_default<
                              typename Handle::parameter_descriptor>> = 0>
  void operator()(Handle &handle) {
    if (!handle.param_set) {
      auto &data = Handle::data_descriptor::get(handle.data);
      data = handle.descriptor.def;
      handle.param_set = true;
    }
  }

  /**
   * \brief Fallback overload, throw an excpetion if paramter is required
   *
   * \tparam Handle Parameter handle type
   *
   * \param handle Parameter handle
   *
   * \note The `required` value must appear to check for non defaulted
   *       missing arguments. Parameters that do not define a default
   *       value and are not marked as required are default
   *       constructed. This is designed so that stackable parameters
   *       (such as sub-parsers) may have an empty list of values.
   */
  template <class Handle, meta::not_use_if<meta::has_default<
                              typename Handle::parameter_descriptor>> = 0>
  void operator()(Handle &handle) {
    if (meta::is_required<typename Handle::parameter_descriptor>::value &&
        !handle.param_set) {
      this->missing_required.push_back(handle.descriptor.flags[0]);
    }
  }

  std::vector<std::string> collective_result() {
    return this->missing_required;
  }
};

/**
 * \brief Iterate over parameters and set unspecified ones to their default
 *
 * \tparam Tuple Handle tuple type
 *
 * \param t Tuple to iterate over
 */
template <class Tuple> std::vector<std::string> fallback_to_default(Tuple &t) {
  return utils::for_each_in_tuple(t, set_default{});
}

/**
 * \brief Matches parameters against the current argument
 */
struct parameter_matcher {
  /// True if a parameter has already matched the current argument
  bool matched = false;

  /**
   * \brief Matches one parameter against the current argument
   *
   * If the parameter decriptor matches, the parameter handle is used
   * to assign a new value.
   *
   * \tparam Handle Parameter handle type
   * \tparam ArgContainer CLI arguments container type
   * \tparam ForwardIt ArgContainer iterator type
   *
   * \param handle Matched parameter handle
   * \param args String container of the CLI
   * \param current_arg Current position in args
   */
  template <typename Handle, class ArgContainer, class ForwardIt, class Parser>
  void operator()(Handle &handle, ArgContainer &args, ForwardIt &current_arg,
                  Parser &p) {
    using std::begin;
    using std::end;

    if (matched || current_arg == end(args)) {
      return;
    }

    std::string current_flag = *current_arg;
    const auto &flags = handle.descriptor.flags;
    matched = find(begin(flags), end(flags), current_flag) != end(flags);

    if (!matched) {
      auto flag_end = current_flag.find_first_of("=");
      if (flag_end != std::string::npos) {

        std::string value_str = current_flag.substr(flag_end + 1);
        current_flag = current_flag.substr(0, flag_end);
        matched = find(begin(flags), end(flags), current_flag) != end(flags);

        if (!meta::is_flagged<typename Handle::parameter_descriptor>::value &&
            matched) {
          *current_arg = current_flag;
          current_arg = args.insert(current_arg + 1, value_str) - 1;
        }
      }
    }

    if (matched) {
      ++current_arg;
      handle.assign(args, current_arg);
      handle.visit(args, current_arg, p);
    }
  }

  /**
   * \brief Returns whether a parameter matched or not
   *
   * \return true if a parameter matched, false otherwise
   */
  bool collective_result() const { return matched; }
};

/**
 * \brief Iterate over parameters to match them against the CLI current argument
 *
 * \tparam Parser Parser type
 * \tparam ArgContainer CLI arguments container type
 * \tparam ForwardIt ArgContainer iterator type
 *
 * \param parser Parser to iterate over
 * \param args String container of the CLI
 * \param current_arg Current position in args
 */
template <class Parser, class ArgContainer, class ForwardIt>
bool match_parameter(Parser &parser, ArgContainer &args,
                     ForwardIt &current_argument) {
  return utils::for_each_in_tuple(parser.handles, parameter_matcher{}, args,
                                  current_argument, parser);
}

/**
 * \brief Validator to be called over every handle
 *
 * This functor is used to validate the parameter descriptors before parsing.
 */
struct parameter_descripor_validator {
  std::unordered_set<std::string> flags;
  std::size_t flag_count;
  std::vector<std::string> missing_flags;

  template <class Handle> void operator()(const Handle &handle) {
    using std::begin;
    using std::end;

    if (detail::size(handle.descriptor.flags) == 0 ||
        (std::find(begin(handle.descriptor.flags), end(handle.descriptor.flags),
                   "") != end(handle.descriptor.flags))) {
      this->missing_flags.emplace_back(typeid(handle.descriptor).name());
    }
    flags.insert(begin(handle.descriptor.flags), end(handle.descriptor.flags));
    flag_count += detail::size(handle.descriptor.flags);
  }

  bool collective_result() const {
    bool flag_conflict = (flag_count != detail::size(flags));
    if (flag_conflict) {
      throw parameter_conflict("Some parameters share the same flags.");
    }
    bool all_named = detail::size(this->missing_flags) == 0;
    if (!all_named) {
      std::stringstream sstr;
      sstr << "Some parameters have no flag defined:";
      for (auto &n : this->missing_flags) {
        sstr << ' ' << n << ',';
      }
      sstr.seekp(-1, std::ios::end);
      sstr << '.';
      throw invalid_parameter(sstr.str());
    }
    return true;
  }
};

/**
 * \brief Run time validation of the descriptors
 */
template <class Tuple> bool validate_descriptors(Tuple &t) {
  return utils::for_each_in_tuple(t, parameter_descripor_validator{});
}

} // namespace detail

template <class... Params> struct parser_descriptor;

/**
 * \brief Parser implementation
 *
 * The parser reads the arguments given through the command line and
 * converts them to values.
 *
 * This implementation implements the basic structure of a parameter
 * descriptor to allow the creation of sub-parsers.
 *
 */
template <class... ParamDescriptors>
class parser : public parser_descriptor<ParamDescriptors...> {

  using handle_list = meta::list<detail::parameter_handle<ParamDescriptors>...>;

public:
  using descriptor = parser_descriptor<ParamDescriptors...>;
  using descriptor::parse;

  /// Parameter handles
  std::tuple<detail::parameter_handle<ParamDescriptors>...> handles;
  std::vector<std::function<bool(const parser &)>> checks;

  /// Default constructor
  parser() = default;

  /**
   * \brief Constructor from descriptors
   *
   * This allows customizing the descriptors before using them, such as
   * changing the default value or some of the associated flags.
   */
  template <class... Ts> parser(Ts &&...descs) {
    auto l = {0,
              ((std::get<handle_list ::template index<
                    detail::parameter_handle<ParamDescriptors>>::value>(handles)
                    .descriptor = std::move(descs)),
               0)...};
    (void)l;
    detail::validate_descriptors(handles);
  }

  /**
   * \brief Run the check functions over the parsed arguments
   */
  void check() {
    for (auto &c : this->checks) {
      c(*this);
    }
  }

  /**
   * \brief Parse the arguments
   *
   * \param args Argument vector
   *
   * \return true if whole command line was parsed, false if an error happened
   */
  bool parse(std::vector<std::string> args) {
    using std::begin;
    using std::end;

    this->program_name = args[0];
    auto current_argument = begin(args) + 1;
    return descriptor::parse(args, current_argument, *this);
  }

  /**
   * \brief Parse the arguments
   *
   * \param argc Argument count
   * \param argv Argument array
   *
   * \return true if whole command line was parsed, false if an error happened
   */
  bool parse(int argc, char **argv) {
    /// List of the arguments passed consterted to std::string
    return this->parse({argv, argv + argc});
  }

  template <class ParamDesc> detail::parameter_handle<ParamDesc> &get_handle() {
    static_assert(
        meta::list<ParamDescriptors...>::template exists<ParamDesc>::value,
        "Parameter class does not exist in given CLI parser.");
    return std::get<handle_list ::template index<
        detail::parameter_handle<ParamDesc>>::value>(handles);
  }

  template <class ParamDesc>
  const detail::parameter_handle<ParamDesc> &get_handle() const {
    return const_cast<parser *>(this)->get_handle<ParamDesc>();
  }

  /**
   * \brief Get given parameter descriptor instance
   *
   * \tparam ParamDesc The parameter descriptor type
   *
   * \return The parameter descriptor instance held by the parser
   */
  template <class ParamDesc> ParamDesc &param() {
    return this->get_handle<ParamDesc>().descriptor;
  }

  /**
   * \brief Get the value(s) for given parameter
   *
   * \tparam ParamDesc The parameter descriptor type
   *
   * \return The data container class held by the parser
   */
  template <class ParamDesc>
  auto get() -> decltype((this->get_handle<ParamDesc>().data)) {
    return this->get_handle<ParamDesc>().data;
  }

  /**
   * \brief Get the value(s) for given parameter
   *
   * \tparam ParamDesc The parameter descriptor type
   *
   * \return The data container class held by the parser
   */
  template <class ParamDesc>
  auto get() const -> decltype((this->get_handle<ParamDesc>().data)) {
    return this->get_handle<ParamDesc>().data;
  }

  /**
   * \brief Get the value(s) for given parameter
   *
   * \param tag Used to deduce the parameter type
   *
   * \tparam ParamDesc The parameter descriptor type
   *
   * \return The data container class held by the parser
   */
  template <class ParamDesc>
  auto get(const ParamDesc &tag)
      -> decltype((this->get_handle<ParamDesc>().data)) {
    (void)tag;
    return this->get_handle<ParamDesc>().data;
  }

  /**
   * \brief Get the value(s) for given parameter
   *
   * \param tag Used to deduce the parameter type
   *
   * \tparam ParamDesc The parameter descriptor type
   *
   * \return The data container class held by the parser
   */
  template <class ParamDesc>
  auto get(const ParamDesc &tag) const
      -> decltype((this->get_handle<ParamDesc>().data)) {
    (void)tag;
    return this->get_handle<ParamDesc>().data;
  }

  /**
   * \brief Check whether option was specified in CLI
   *
   * \tparam ParamDesc The parameter descriptor
   *
   * \return true if the parameter was seen in the argument list given to
   * the parser
   */
  template <class ParamDesc> bool exists() const {
    return this->get_handle<ParamDesc>().param_set;
  }

  /**
   * \brief Formatted output operator for debugging
   */
  friend std::ostream &operator<<(std::ostream &os, const parser &p) {
    os << p.handles;
    return os;
  }
};

template <class... Params> struct parser_descriptor {
  /// Program name
  std::string program_name;
  /// Program description
  std::string program_description;

  /// Parameter descriptor type
  using type = parser<Params...>;
  /// Allow several sub parser instances to appear in the parameter string
  enum { flagged, tcli_parser };

  /**
   * \brief Parse the arguments starting from given position
   *
   * \param args Arguments vector
   * \param current_arg Iterator to an element of args from where to start
   *        parsing
   *
   * \return true if whole command line was parsed, false if an error happened
   */
  template <class Container, class It>
  static bool parse(Container &args, It &current_arg, parser<Params...> &p) {
    bool matched = false;
    while (current_arg != end(args)) {
      matched = false;

      try {
        matched = detail::match_parameter(p, args, current_arg);
      } catch (parse_error &e) { // sub-parser end of parse
        if (e.unknown_parameter && !e.missing_required) {
          matched = true;
        } else {
          throw e;
        }
      }

      if (!matched) {
        break;
      }
    }

    auto missing = fallback_to_default(p.handles);
    p.check();

    std::stringstream sstr;
    std::pair<bool, bool> err;
    if (!matched && current_arg != end(args)) {
      sstr << ("Unknown argument: '" + *current_arg + "'\n'");
      err.first = true;
    }

    if (!missing.empty()) {
      sstr << "Required parameters ";
      for (auto &f : missing) {
        sstr << f << ", ";
      }
      sstr << " were not set and have no default.\n";
      err.second = true;
    }

    if (err.first || err.second) {
      parse_error ex{sstr.str()};
      ex.unknown_parameter = err.first;
      ex.missing_required = err.second;
      throw ex;
    }

    return true;
  }
};

/**
 * \brief Mandatory base class for TCLI tags
 */
struct tag {};

/** \brief Base class to add `stackable` tag */
struct stackable_tag : tag {
  enum { stackable };
};

/** \brief Base class to add `required` tag */
struct required_tag : tag {
  enum { required };
};

struct flag_tag : tag {
  enum { flagged };
  using type = bool;
};

struct group {};

namespace detail {
struct build_now {};
} // namespace detail

/** When the build_now tag is the first parameter, all the parameters have
 * been processed, build the parser */
template <class... Params>
auto build_parser(detail::build_now, Params &&...params)
    -> parser<typename std::decay<Params>::type...> {
  return {params...};
}

/** implementation: Explode a parameter group into the parameters it contains */
template <class Group, class... Params, std::size_t... Is>
auto explode_group(std::index_sequence<Is...>, Group &&g, Params &&...ps)
    -> decltype(build_parser(std::get<Is>(std::forward<Group>(g))...,
                             std::forward<Params>(ps)...)) {
  return build_parser(std::get<Is>(std::forward<Group>(g))...,
                      std::forward<Params>(ps)...);
}

/** Explode a parameter group into the parameters it contains */
template <class... Ts, class... Params>
auto explode_group(const std::tuple<Ts...> &g, Params &&...ps)
    -> decltype(explode_group(std::index_sequence_for<Ts...>{}, g,
                              std::forward<Params>(ps)...)) {
  return explode_group(std::index_sequence_for<Ts...>{}, g,
                       std::forward<Params>(ps)...);
}

/** When a constrained group appears, split it up and add its constraint to the
 * resulting parser */
template <
    class Group, class... Ts, class... Params,
    meta::use_if<std::is_base_of<group, typename std::decay<Group>::type>> = 0>
auto build_parser(Group &&g, Params &&...ps)
    -> decltype(explode_group(std::forward<Group>(g),
                              std::forward<Params>(ps)...)) {
  auto p = explode_group(std::forward<Group>(g), std::forward<Params>(ps)...);
  p.checks.emplace_back(typename std::decay<Group>::type::constraint{});
  return p;
}

template <class... Ts> struct get_parser_type;

/** A normal parameter is pushed to the end of the parameter list */
template <
    class T, class... Params,
    meta::not_use_if<
        std::is_same<typename std::decay<T>::type, detail::build_now>> = 0,
    meta::not_use_if<std::is_base_of<group, typename std::decay<T>::type>> = 0>
auto build_parser(T &&p, Params &&...ps) ->
    typename get_parser_type<Params..., T>::type {
  return build_parser(std::forward<Params>(ps)..., std::forward<T>(p));
}

template <class... Ts> struct get_parser_type {
  using type = decltype(build_parser(std::declval<Ts>()...));
};

/** Call the build_parser functions, adds the build_now tag at the end of the
 * parameter list
 */
template <class... Params>
auto make_parser(Params &&...ps)
    -> decltype(build_parser(std::forward<Params>(ps)...,
                             detail::build_now{})) {
  return build_parser(std::forward<Params>(ps)..., detail::build_now{});
}

template <class... Params> struct check_exclusive {
  template <class T> bool operator()(const T &p) {
    std::size_t count = 0;
    auto l = {0, ((count += p.template exists<Params>()), 0)...};
    (void)l;

    if (count > 1) {
      throw exclusive_parameters("Incompatible parameters defined");
    }

    return true;
  }
};

template <class... Ts> struct _xor : std::tuple<Ts...>, group {
  using constraint = check_exclusive<Ts...>;
  using std::tuple<Ts...>::tuple;
};

template <class... Ts>
_xor<typename std::decay<Ts>::type...> xor_group(Ts &&...ts) {
  return _xor<typename std::decay<Ts>::type...>(std::forward<Ts>(ts)...);
}

using str_vec = std::vector<std::string>;
using flag_list = std::vector<std::string>;

} // namespace cpp_tools::cl_parser

#endif /* TCLI_HPP */
