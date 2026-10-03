// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tools/bench.hpp
// --------------------------------
#pragma once

#include <cpp_tools/timers/simple_timer.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <unordered_map>

namespace scalfmm::bench
{
    /**
     * @brief
     *
     * @tparam DurationType
     * @param timers
     * @return auto
     */
    template<typename DurationType>
    inline auto compute(std::unordered_map<std::string, cpp_tools::timers::timer<DurationType>> timers)
    {
        using duration_type = DurationType;
        using value_type = double;
        static constexpr value_type unit_multiplier = static_cast<value_type>(
          duration_type::period::den);   // denominator of the ratio: milli = 10^3, micro = 10^6, nano = 10^9

        value_type overall{0.};

        for(auto e: timers)
        {
            overall += value_type(e.second.elapsed()) / unit_multiplier;
        }
        auto neartime = value_type(timers["p2p"].elapsed()) / unit_multiplier;
        auto fartime = value_type(timers["p2m"].elapsed() + timers["m2m"].elapsed() + timers["m2l"].elapsed() +
                                  timers["field0"].elapsed() + timers["l2l"].elapsed() + timers["l2p"].elapsed()) /
                       unit_multiplier;
        auto ratio{value_type(fartime) / value_type(neartime)};

        return std::make_tuple(fartime, neartime, overall, ratio);
    }

    /**
     * @brief
     *
     * @tparam DurationType
     * @param timers
     * @return auto
     */
    template<typename DurationType>
    inline auto print(std::unordered_map<std::string, cpp_tools::timers::timer<DurationType>> timers)
    {
        using duration_type = DurationType;
        using value_type = double;
        static constexpr value_type unit_multiplier = static_cast<value_type>(
          duration_type::period::den);   // denominator of the ratio: milli = 10^3, micro = 10^6, nano = 10^9

        auto [fartime, neartime, overall, ratio] = compute(timers);

        std::cout << "[time][leaf_to_cell pass] : " << value_type(timers["p2m"].elapsed()) / unit_multiplier << "\n";
        std::cout << "[time][upward pass]       : " << value_type(timers["m2m"].elapsed()) / unit_multiplier << "\n";
        std::cout << "[time][transfer pass]     : " << value_type(timers["m2l"].elapsed()) / unit_multiplier << "\n";
        std::cout << "[time][downward pass]     : " << value_type(timers["l2l"].elapsed()) / unit_multiplier << "\n";
        std::cout << "[time][cell_to_leaf pass] : " << value_type(timers["l2p"].elapsed()) / unit_multiplier << "\n";
        std::cout << "[time][direct pass]       : " << value_type(timers["p2p"].elapsed()) / unit_multiplier << "\n";
        std::cout << "[time][field0 pass]       : " << value_type(timers["field0"].elapsed()) / unit_multiplier << "\n";
        std::cout << "[time][m2l list]          : " << value_type(timers["m2l-list"].elapsed()) / unit_multiplier
                  << "\n";
        std::cout << "[time][p2p list]          : " << value_type(timers["p2p-list"].elapsed()) / unit_multiplier
                  << "\n";
        std::cout << "[time][far time]   : " << fartime << '\n';
        std::cout << "[time][near time]  : " << neartime << '\n';
        std::cout << "[time][diff time]  : " << std::abs(fartime - neartime) << '\n';
        std::cout << "[time][ratio time] : " << ratio << '\n';
        std::cout << "[time][full algo]  : " << overall << '\n';
        return std::make_tuple(fartime, neartime, overall, ratio);
    }

    /**
     * @brief
     *
     * @tparam String
     * @tparam Strings
     * @param file_name
     * @param header
     * @param arg
     * @param args
     */
    template<typename String, typename... Strings>
    inline auto dump_csv(std::string file_name, std::string header, String arg, Strings... args) -> void
    {
        std::ofstream benchfile;
        if(std::filesystem::exists(file_name))
        {
            benchfile.open(file_name, std::ios::app);
        }
        else
        {
            benchfile.open(file_name);
            benchfile << header << '\n';
        }
        benchfile << arg;
        (benchfile << ... << (',' + args));
        benchfile << '\n';
        benchfile.close();
    }
}   // namespace scalfmm::bench
