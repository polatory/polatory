#ifndef _CHECKER_HPP_
#define _CHECKER_HPP_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>

namespace inria {

struct locus_t {
    int line;
    std::string file;

    friend std::ostream& operator<<(std::ostream& os, const locus_t& l) {
        return os << l.file << ':' << l.line;
    }

    friend locus_t operator+(const locus_t& l, int offset) {
        return {l.line + offset, l.file};
    }
};

#define LOCUS ::inria::locus_t{__LINE__,__FILE__}

class checker_t {

    template<class T>
    struct has_formated_output {
        template<class U>
        static constexpr auto check(std::ostream* os, U* o)
            -> decltype((*os << *o), std::true_type{} ) ;
        static constexpr auto check(...)
            -> std::false_type;
        enum {value = decltype(check(nullptr, (T*)0))::value};
    };

    struct res_t {
        bool val;
        locus_t locus;

        operator bool() const noexcept {return val;}
    };

    bool silence_;
    std::vector<res_t> results;

    template<class T, class = typename std::enable_if<has_formated_output<T>::value>::type>
    const T& get_print_val(const T& t) {
        return t;
    }

    template<class T, class = typename std::enable_if<! has_formated_output<T>::value>::type>
    std::string get_print_val(const T&) {
        return "<non-printable object>";
    }

    template<class T, class U>
    bool check_res(bool res, std::string desc, const T& value, const U& target, locus_t locus) {
        results.emplace_back(res_t{res, locus});
        if(! results.back()) {
            if(locus.line) {
                std::cerr << locus << ": ";
            }
            std::cerr << "Check failed (id: " << results.size() << "): "
                      << get_print_val(value)
                      << " <" << desc << "> "
                      << get_print_val(target)
                      << '\n';
        }
        return results.back();
    }

public:
    checker_t() = default;
    checker_t(const checker_t&) = default;
    checker_t(checker_t&&) = default;
    checker_t& operator=(const checker_t&) = default;
    checker_t& operator=(checker_t&&) = default;

    checker_t(bool silence) : silence_(silence) {}

    bool silence() const {
        return silence_;
    }
    bool silence(bool value) {
        return silence_ = value;
    }

    bool fail(const std::string& msg, locus_t locus) {
        return check_res(false, msg, "", "", locus);
    }

    bool succeed(locus_t locus) {
        return check_res(true, "", "", "", locus);
    }

    template<class T>
    bool is_true(const T& value, locus_t locus) {
        return equal(value, true, locus);
    }

    template<class T>
    bool is_false(const T& value, locus_t locus) {
        return equal(value, false, locus);
    }

    template<class T, class U>
    bool equal(const T& value, const U& target, locus_t locus) {
        return check_res(value == target, "equal", value, target, locus);
    }

    template<class T, class U>
    bool different(const T& value, const U& target, locus_t locus) {
        return check_res(value != target, "different", value, target, locus);
    }

    template<class T, class U>
    bool less_eq(const T& value, const U& target, locus_t locus) {
        return check_res(value <= target, "less_eq", value, target, locus);
    }

    template<class T, class U>
    bool less(const T& value, const U& target, locus_t locus) {
        return check_res(value < target, "less", value, target, locus);
    }

    template<class T, class U>
    bool greater(const T& value, const U& target, locus_t locus) {
        return check_res(value > target, "greater", value, target, locus);
    }

    template<class T, class U>
    bool greater_eq(const T& value, const U& target, locus_t locus) {
        return check_res(value >= target, "greater_eq", value, target, locus);
    }

    std::pair<std::size_t, std::size_t> summary() const {
        return {std::count(std::begin(results),std::end(results),true),
                results.size()};
    }

    bool ok() const {
        return std::all_of(std::begin(results), std::end(results), [](char c) {return c;});
    };

    void print_summary() {
        std::size_t passed, tried;
        std::tie(passed, tried) = summary();
        std::size_t failed = tried - passed;
        if(passed == tried) {
            std::cout << "Tests PASSED (" << passed << ")\n";
        } else {
            std::cout << failed << '/' << tried << " test" << (failed > 1 ? "s" : "") << " failed.\n";
        }
    }

    ~checker_t() {
        if(!this->silence() && results.size() != 0) {
            print_summary();
        }
    }
};

} // end namespace inria

#endif /* _CHECKER_HPP_ */
