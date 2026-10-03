// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/accurater.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_ACCURATER_HPP
#define SCALFMM_UTILS_ACCURATER_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace scalfmm
{
    namespace utils
    {
        /**
         * @brief A class to compute accuracy  between data.
         *
         * @tparam ValueType
         */
        template<class ValueType>
        class accurater
        {
            using value_type = ValueType;

            /**
             * @brief Number of elements used to compute the error.
             *
             */
            std::size_t _nb_elements{};

            /**
             * @brief l2-norm of the reference value.
             *
             */
            value_type _l2_norm{};

            /**
             * @brief l2-norm of the difference between the reference value and the value.
             *
             */
            value_type _l2_diff{};

            /**
             * @brief Infinity norm of the reference value.
             *
             */
            value_type _max{};

            /**
             * @brief Infinity norm of the difference between the reference value and the value.
             *
             */
            value_type _max_diff{};

          public:
            /**
             * @brief Construct a new accurater object
             *
             */
            accurater() = default;

            /**
             * @brief Construct a new accurater object
             *
             */
            accurater(accurater const&) = default;

            /**
             * @brief Construct a new accurater object
             *
             */
            accurater(accurater&&) noexcept = default;

            /**
             * @brief
             *
             * @return accurater&
             */
            inline auto operator=(accurater const&) -> accurater& = default;

            /**
             * @brief
             *
             * @return accurater&
             */
            inline auto operator=(accurater&&) noexcept -> accurater& = default;

            /**
             * @brief Destroy the accurater object
             *
             */
            ~accurater() = default;

            /**
             * @brief Construct a new accurater object with initial values.
             *
             * @param ref_value
             * @param value
             * @param nbValues
             */
            accurater(const value_type ref_value[], const value_type value[], const std::size_t& nbValues)
            {
                this->add(ref_value, value, nbValues);
            }

            /**
             * @brief Add |ref_value-value| to the current accurater
             *
             * @param ref_value
             * @param value
             */
            inline auto add(const value_type& ref_value, const value_type& value) -> void
            {
                _l2_diff += (value - ref_value) * (value - ref_value);
                _l2_norm += ref_value * ref_value;
                _max = std::max(_max, std::abs(ref_value));
                _max_diff = std::max(_max_diff, std::abs(ref_value - value));
                ++_nb_elements;
            }

            /**
             * @brief Add all vector elements |ref_value[i]-value[i]| to the current accurater
             *
             * @param ref_value reference array
             * @param value  array to check
             * @param nb_values
             */
            inline auto add(const value_type* const ref_values, const value_type* const values,
                            const std::size_t& nb_values) -> void
            {
                for(std::size_t idx = 0; idx < nb_values; ++idx)
                {
                    this->add(ref_values[idx], values[idx]);
                }
                _nb_elements += nb_values;
            }

            /**
             * @brief
             *
             * @tparam VectorType
             * @param ref_values
             * @param values
             */
            template<class VectorType>
            inline auto add(const VectorType& ref_values, const VectorType& values) -> void
            {
                if(values->size() != ref_values->size())
                {
                    std::cerr << "Wrong size in add method. " << ref_values.size() << " != " << values << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                this->add(ref_values->data(), values->data(), values->size());
            }

            /**
             * @brief Add an accurater
             *
             * @param inAcc
             */
            inline auto add(const accurater& inAcc) -> void
            {
                _l2_diff += inAcc.get_l2_diff_norm();
                _l2_norm += inAcc.get_l2_norm();
                _max = std::max(_max, inAcc.get_ref_max());
                _max_diff = std::max(_max_diff, inAcc.get_infinity_norm());
                _nb_elements += inAcc.get_nb_elements();
            }

            /**
             * @brief Get the the square of the l2 norm of the error sum_i(ref_i-val_i)^2)
             *
             * @return value_type
             */
            inline auto get_l2_diff_norm() const -> value_type { return _l2_diff; }

            /**
             * @brief Get the l2 diff norm2 object
             *
             * @return value_type
             */
            inline auto get_l2_diff_norm2() const -> value_type { return _l2_diff; }

            /**
             * @brief Get the l2 norm of the error sqrt( sum_i(ref_i-val_i)^2)
             *
             * @return value_type
             */
            inline auto get_l2_norm() const -> value_type { return std::sqrt(_l2_diff); }

            /**
             * @brief Get the l2 norm of the reference vector sqrt( sum_i(ref_i)^2)
             *
             * @return value_type
             */
            inline auto get_ref_l2_norm() const -> value_type { return std::sqrt(_l2_norm); }

            /**
             * @brief Get the maximum value of the reference max_i(|ref_i|)
             *
             * @return value_type
             */
            inline auto get_ref_max() const -> value_type { return _max; }

            /**
             * @brief Get the nb elements used in the norm computation
             *
             * @return auto
             */
            inline auto get_nb_elements() const { return _nb_elements; }

            /**
             * @brief Set the nb elements object
             *
             * @param n
             */
            inline auto set_nb_elements(const std::size_t& n) -> void { _nb_elements = n; }

            /**
             * @brief Get the mean squared error object.
             *
             * @return value_type
             */
            inline auto get_mean_squared_error() const -> value_type { return std::sqrt(_l2_diff); }

            /**
             * @brief Get the root-mean-square error.
             *
             * @return value_type
             */
            inline auto get_rms_error() const -> value_type
            {
                return std::sqrt(_l2_diff / static_cast<value_type>(_nb_elements));
            }

            /**
             * @brief Get the infinity norm of the error max_i|ref_i-val_i|
             *
             * @return value_type
             */
            inline auto get_infinity_norm() const -> value_type { return _max_diff; }

            /**
             * @brief Get the relative L2 norm of the error  sqrt( sum_i(ref_i-val_i)^2/sum_i(ref_i^2))
             *
             * @return value_type
             */
            inline auto get_relative_l2_norm() const -> value_type { return std::sqrt(_l2_diff / _l2_norm); }

            /**
             * @brief Get the relative infinity norm of the error  max_i|ref_i-val_i|/max_i|ref_i|
             *
             * @return value_type
             */
            inline auto get_relative_infinity_norm() const -> value_type { return _max_diff / _max; }

            /**
             * @brief Print the relative errors (L2, RMS, Infinity)
             *
             * @tparam StreamClass
             * @param output  the stream
             * @param inAccurater  the accurater containing the errors to print
             * @return StreamClass&
             */
            template<typename StreamClass>
            inline friend auto operator<<(StreamClass& output, const accurater& inAccurater) -> StreamClass&
            {
                output << "[Error] Relative L2-norm = " << inAccurater.get_relative_l2_norm()
                       << " \t RMS norm = " << inAccurater.get_rms_error()
                       << " \t Relative  infinity norm = " << inAccurater.get_relative_infinity_norm();
                return output;
            }

            /**
             * @brief reset the accurater
             *
             */
            inline auto reset() -> void
            {
                _l2_norm = value_type(0);
                _l2_diff = value_type(0);
                _max = value_type(0);
                _max_diff = value_type(0);
                _nb_elements = 0;
            }
        };

    }   // namespace utils
}   // namespace scalfmm

#endif   // SCALFMM_UTILS_ACCURATER_HPP
