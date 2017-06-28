#if __cplusplus < 201402L

#ifndef _SCALFMM_MAKE_UNIQUE_HPP_
#define _SCALFMM_MAKE_UNIQUE_HPP_

#include <utility>
#include <memory>

namespace std {
    template<class T, class... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

#endif
#endif
