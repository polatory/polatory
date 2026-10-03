// --------------------------------
// See LICENCE file at project root
// File : scalfmm/tools/progress_bar.hpp
// --------------------------------
#ifndef SCALFMM_TOOLS_PROGRESS_BAR_HPP
#define SCALFMM_TOOLS_PROGRESS_BAR_HPP

#include "scalfmm/tools/colorized.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace scalfmm::tools
{
    /**
     * @brief
     *
     * @tparam Object
     * @param obj
     * @return int
     */
    template<typename Object>
    int progress(Object& obj);

    /**
     * @brief
     *
     */
    struct progress_bar
    {
        /**
         * @brief
         *
         */
        std::stringstream sstr{};

        /**
         * @brief
         *
         */
        std::thread t;

        /**
         * @brief
         *
         * @tparam Object
         * @param obj
         */
        template<typename Object>
        void follow(Object& obj)
        {
            this->t = std::thread(
              [this, &obj]()
              {
                  bool run = true;
                  while(run)
                  {
                      sstr.str("");
                      sstr.clear();
                      sstr.precision(4);
                      int p = progress(obj);
                      std::cout << p << std::endl;
                      sstr << "[";
                      for(int i = 0; i < 100; ++i)
                      {
                          sstr << (i < p ? "\u2038" : " ");
                      }
                      sstr << "] " << p << "%    ";
                      std::cout << '\r' << sstr.str() << std::flush;
                      run = p < 100;
                      std::this_thread::sleep_for(std::chrono::milliseconds(200));
                  }
              });
        }

        /**
         * @brief
         *
         */
        void finish() { this->t.join(); }
    };
}   // namespace scalfmm::tools

#endif
