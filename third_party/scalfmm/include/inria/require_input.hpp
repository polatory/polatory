#ifndef REQUIRE_INPUT_HPP
#define REQUIRE_INPUT_HPP

#include <istream>

namespace inria {


    struct require_input {
        const char* reference;

        require_input(const char* in) : reference(in) {}

        friend std::istream& operator>>(std::istream& is, const require_input& ci) {
            char c;
            for(const char* p = ci.reference; *p != '\0'; ++p) {
                if(std::isspace(*p)) {
                    std::istream::sentry sentry(is);
                    if(! sentry) {
                        return is;
                    }
                    continue;
                }
                if((c = is.peek()) != *p) {
                    is.setstate(std::ios::failbit);
                    return is;
                }
                is.get();
            }
            return is;
        }
    };

}



#endif /* REQUIRE_INPUT_HPP */
