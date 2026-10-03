// --------------------------------
// See LICENCE file at project root
// File : scalfmm/algorithms/common.hpp
// --------------------------------

#ifndef SCALFMM_ALGORITHMS_COMMON_HPP
#define SCALFMM_ALGORITHMS_COMMON_HPP

#include <cmath>
#include <iostream>
#include <string>

namespace scalfmm::algorithms
{
    /**
    * @brief Specifies the operations to perform in the algorithm.
    */
    enum operators_to_proceed : unsigned int
    {
        p2p = (1 << 0),    ///< Particles to Particles operator (Near field)
        p2m = (1 << 1),    ///< Particles to Multipole operator (Far field)
        m2m = (1 << 2),    ///< Multipole to Multipole operator (Far field)
        m2l = (1 << 3),    ///< Multipole to Local operator     (Far field)
        l2l = (1 << 4),    ///< Local to Local operator         (Far field)
        l2p = (1 << 5),    ///< Local to Particles operator     (Far field)
        p2l = (1 << 6),    ///< Particles to Local operator     (Far field in Adaptive algorithm)
        m2p = (1 << 7),    ///< Multipole to Particles operator (Far field in Adaptive algorithm)
        nearfield = p2p,   ///< Near field operator
        farfield = (p2m | m2m | m2l | l2l | l2p | p2l | p2l | m2p),   ///< Only Far Field operators
        all = (nearfield | farfield)                                  ///< Near and far field operators
    };

    /**
    * @brief Converts operator code to string.
    *
    * @param op operator code.
    *
    * @return string corresponding to the operator code.
    */
    [[nodiscard]] inline auto to_string(operators_to_proceed op) -> std::string
    {
        std::string s("");
        switch(op)
        {
        case operators_to_proceed::farfield:
            s = "farfield";
            break;
        case operators_to_proceed::all:
            s = "full";
            break;
        case operators_to_proceed::p2p:
            s = "p2p";
            break;
        case operators_to_proceed::p2m:
            s = "p2m";
            break;
        case operators_to_proceed::m2m:
            s = "m2m";
            break;
        case operators_to_proceed::m2l:
            s = "m2l";
            break;
        case operators_to_proceed::l2l:
            s = "l2l";
            break;
        case operators_to_proceed::l2p:
            s = "l2p";
            break;
        case operators_to_proceed::p2l:
            s = "p2l";
            break;
        case operators_to_proceed::m2p:
            s = "m2p";
            break;
        }
        return s;
    };

    /**
    * @brief builds string from operator code
    *
    * @param op
    *
    * @return
    */
    [[nodiscard]] inline auto build_string(const unsigned int op) -> std::string
    {
        std::string s("");
        unsigned int tmp{op};
        while(tmp > 1)
        {
            unsigned int top = static_cast<unsigned int>(std::log2(tmp));
            auto ope = static_cast<unsigned int>(1 << top);
            tmp -= 1 << top;
            operators_to_proceed oop{ope};
            s = "|" + to_string(oop) + s;
        }
        if(tmp == 1)
        {
            s += "p2p";
        }
        else
        {
            s.erase(0, 1);
        }
        return s;
    };

    /**
    * @brief prints string corresponding to the operator code.
    *
    * @param op
    */
    inline void print(const unsigned int op) { std::cout << build_string(op); };

    /**
    * @brief prints the operator code.
    *
    * @param op
    */
    inline void print(const operators_to_proceed op)
    {
        switch(op)
        {
        case operators_to_proceed::farfield:
            std::cout << "farfield";
            break;
        case operators_to_proceed::all:
            std::cout << "full";
            break;
        case operators_to_proceed::p2p:
            std::cout << "p2p";
            break;
        case operators_to_proceed::p2m:
            std::cout << "p2m";
            break;
        case operators_to_proceed::m2m:
            std::cout << "m2m";
            break;
        case operators_to_proceed::m2l:
            std::cout << "m2l";
            break;
        case operators_to_proceed::l2l:
            std::cout << "l2l";
            break;
        case operators_to_proceed::l2p:
            std::cout << "l2p";
            break;
        case operators_to_proceed::p2l:
            std::cout << "p2l";
            break;
        case operators_to_proceed::m2p:
            std::cout << "m2p";
            break;
        }
    };

}   // namespace scalfmm::algorithms

#endif   // SCALFMM_ALGORITHMS_COMMON_HPP
