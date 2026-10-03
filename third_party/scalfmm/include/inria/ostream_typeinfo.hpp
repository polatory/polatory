#ifndef OSTREAM_TYPEINFO_HPP
#define OSTREAM_TYPEINFO_HPP

#include <ostream>

#include <typeinfo>
#include <unordered_map>
#include <cxxabi.h>

#include <iostream>

namespace inria {

namespace details {

template<class T>
struct info_ {
    using type = T;
};


template<class T>
std::ostream& operator<<(std::ostream& os, info_<T>) {
    static std::unordered_map<std::string, std::string> map;

    const std::type_info& ti = typeid(info_<T>);

    if(map.end() == map.find(ti.name())) {
        int status = 0;
        char* realname = NULL;
        realname = abi::__cxa_demangle(ti.name(), 0, 0, &status);
        std::string name {realname};
        auto b = name.find_first_of('<');
        auto e = name.find_last_of('>');
        map.emplace(std::pair<std::string, std::string>{ti.name(), name.substr(b+1, e-b-1)});
        free(realname);
    }

    os << map.at(ti.name());

    return os;
}

}

template<class T>
details::info_<T> type_info(T&&) {
    return {};
}

template<class T>
details::info_<T> type_info() {
    return {};
}

}

#endif /* OSTREAM_TYPEINFO_HPP */
