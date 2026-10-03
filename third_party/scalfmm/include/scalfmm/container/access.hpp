// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/access.hpp
// --------------------------------
#pragma once

#include "scalfmm/meta/utils.hpp"

#include <cpp_tools/colors/colorized.hpp>
#include <utility>

namespace scalfmm::container
{
    /**
     * @brief A lazy light iterator on a sub tuple on an particle iterator
     *
     * The lazy iterator is composed of
     *  - a tuple of iterator pointing on the begin
     *  - an index to access to the true element
     *  If you want to manipulate the element you have to evaluate the light iterator
     *   by dereferencing
     *
     * @tparam TuplePtr a tuple of type of pointer
     * @tparam IsConst to specify if the iterator is constant or nat
     */
    template<typename TuplePtr, bool IsConst>
    struct light_tuple_iterator
    {
        using tuple_type = meta::replace_inner_tuple_type_t<std::remove_pointer_t, TuplePtr>;
        using tuple_ptr_type = meta::replace_inner_tuple_type_t<std::decay_t, TuplePtr>;
        using tuple_ptr_const_type = meta::replace_inner_tuple_type_t<std::add_const_t, tuple_ptr_type>;
        using tuple_ref_type = meta::replace_inner_tuple_type_t<std::add_lvalue_reference_t, tuple_type>;
        using tuple_const_ref_type = meta::replace_inner_tuple_type_t<std::add_const_t, tuple_ref_type>;

        // Need to adapt iterator category !!!
        using iterator_category = std::random_access_iterator_tag;
        using value_type = tuple_type;
        using reference = std::conditional_t<IsConst, tuple_const_ref_type, tuple_ref_type>;
        using pointer = void;
        using difference_type = std::size_t;

        static constexpr bool is_const_qualified{IsConst};

      private:
        /**
         * @brief
         *
         */
        tuple_ptr_const_type vec_{};

        /**
         * @brief
         *
         */
        int index_{0};

      public:
        /**
         * @brief Construct a new light tuple iterator object
         *
         */
        light_tuple_iterator() = default;

        /**
         * @brief Construct a new light tuple iterator object
         *
         */
        light_tuple_iterator(light_tuple_iterator const&) = default;

        /**
         * @brief Construct a new light tuple iterator object
         *
         */
        light_tuple_iterator(light_tuple_iterator&&) noexcept = default;

        /**
         * @brief
         *
         * @return light_tuple_iterator&
         */
        inline auto operator=(light_tuple_iterator const&) -> light_tuple_iterator& = default;

        /**
         * @brief
         *
         * @return light_tuple_iterator&
         */
        inline auto operator=(light_tuple_iterator&&) noexcept -> light_tuple_iterator& = default;

        /**
         * @brief Destroy the light tuple iterator object
         *
         */
        ~light_tuple_iterator() = default;

        /**
         * @brief Construct a new light tuple iterator object
         *
         * @param vec
         * @param index
         */
        light_tuple_iterator(tuple_ptr_const_type vec, int index) noexcept
          : vec_{vec}
          , index_{index}
        {
        }

        /**
        * @brief Construct a new light tuple iterator object
        *
        * @tparam T
        * @param other
        */
        template<typename T>
        light_tuple_iterator(light_tuple_iterator<T, not(IsConst)> const& other) noexcept
          : vec_{other.data()}
          , index_{other.index()}
        {
        }

        /**
         * @brief
         *
         * @param os
         * @param iter
         * @return std::ostream&
         */
        inline friend auto operator<<(std::ostream& os, light_tuple_iterator iter) -> std::ostream&
        {
            auto tup = iter.data();
            auto index = iter.index();

            auto print_tuple = [&os](auto const& tuples)
            {
                os << "[";
                meta::for_each(tuples, [&os](auto const& v) { os << v << ", "; });
                os << "] ";
            };
            auto print_tuple_ptr = [&os](auto const& tuples)
            {
                os << "[";
                meta::for_each(tuples, [&os](auto const& v) { os << &v << ", "; });
                os << "]";
            };
            os << cpp_tools::colors::red;
            os << " proxy: ";
            print_tuple_ptr(*iter);
            os << "=  (";
            print_tuple(tup);
            os << ") index: " << index;
            os << cpp_tools::colors::reset;
            return os;
        }

      private:
        /**
         * @brief
         *
         * @tparam Is
         * @param s
         * @return auto
         */
        template<size_t... Is>
        [[nodiscard]] inline auto make_proxy(std::index_sequence<Is...> s) const noexcept
        {
            return std::forward_as_tuple(std::get<Is>(vec_)[index_]...);
        }

      public:
        /**
         * @brief
         *
         * @return auto
         */
        [[nodiscard]] inline auto operator*() const noexcept
        {
            return make_proxy(std::make_index_sequence<std::tuple_size_v<tuple_type>>{});
        }

        /**
         * @brief
         *
         * @return tuple_ptr_type
         */
        [[nodiscard]] inline auto data() const noexcept -> tuple_ptr_type { return this->vec_; }

        /**
         * @brief
         *
         * @return int
         */
        [[nodiscard]] inline auto index() const noexcept -> int { return this->index_; }

        /**
         * @brief
         *
         * @param rhs
         * @return true
         * @return false
         */
        [[nodiscard]] inline auto operator==(light_tuple_iterator const& rhs) const noexcept -> bool

        {
            return index_ == rhs.index_;
        }

        /**
         * @brief
         *
         * @param rhs
         * @return true
         * @return false
         */
        [[nodiscard]] inline auto operator!=(light_tuple_iterator const& rhs) const noexcept -> bool
        {
            return !(*this == rhs);
        }

        /**
         * @brief
         *
         * @param rhs
         * @return true
         * @return false
         */
        [[nodiscard]] inline auto operator<(light_tuple_iterator const& rhs) const noexcept -> bool
        {
            return index_ < rhs.index_;
        }

        /**
         * @brief
         *
         * @param rhs
         * @return true
         * @return false
         */
        [[nodiscard]] inline auto operator>(light_tuple_iterator const& rhs) const noexcept -> bool
        {
            return rhs < *this;
        }

        /**
         * @brief
         *
         * @param rhs
         * @return true
         * @return false
         */
        [[nodiscard]] inline auto operator<=(light_tuple_iterator const& rhs) const noexcept -> bool
        {
            return !(rhs < *this);
        }

        /**
         * @brief
         *
         * @param rhs
         * @return true
         * @return false
         */
        [[nodiscard]] inline auto operator>=(light_tuple_iterator const& rhs) const noexcept -> bool
        {
            return !(*this < rhs);
        }

        /**
         * @brief
         *
         * @return light_tuple_iterator&
         */
        inline auto operator++() noexcept -> light_tuple_iterator& { return ++index_, *this; }

        /**
         * @brief
         *
         * @return light_tuple_iterator&
         */
        inline auto operator--() noexcept -> light_tuple_iterator& { return --index_, *this; }

        /**
         * @brief
         *
         * @return light_tuple_iterator
         */
        inline auto operator++(int) noexcept -> light_tuple_iterator
        {
            const auto old = *this;
            return ++index_, old;
        }

        /**
         * @brief
         *
         * @return light_tuple_iterator
         */
        inline auto operator--(int) noexcept -> light_tuple_iterator
        {
            const auto old = *this;
            return --index_, old;
        }

        /**
         * @brief
         *
         * @param shift
         * @return light_tuple_iterator&
         */
        inline auto operator+=(int shift) noexcept -> light_tuple_iterator& { return index_ += shift, *this; }

        /**
         * @brief
         *
         * @param shift
         * @return light_tuple_iterator&
         */
        inline auto operator-=(int shift) noexcept -> light_tuple_iterator& { return index_ -= shift, *this; }

        /**
         * @brief
         *
         * @param shift
         * @return light_tuple_iterator
         */
        inline auto operator+(int shift) const noexcept -> light_tuple_iterator
        {
            return light_tuple_iterator(vec_, index_ + shift);
        }

        /**
         * @brief
         *
         * @param shift
         * @return light_tuple_iterator
         */
        inline auto operator-(int shift) const noexcept -> light_tuple_iterator
        {
            return light_tuple_iterator(vec_, index_ - shift);
        }

        /**
         * @brief
         *
         * @param rhs
         * @return int
         */
        inline auto operator-(light_tuple_iterator const& rhs) const noexcept -> int { return index_ - rhs.index_; }
    };

    /**
     * @brief Return an interator on the particle
     *
     * @tparam ProxyParticleIterator
     * @param
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto begin(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        return p.first;
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto end(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        return p.second;
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto position_begin(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        using range_type = typename ProxyParticleIterator::range_position_type;
        using sub_tuple_type = std::decay_t<decltype(meta::sub_tuple(p.first.data(), range_type{}))>;
        using iterator_type = light_tuple_iterator<sub_tuple_type, ProxyParticleIterator::is_const_qualified>;
        return iterator_type(meta::make_sub_tuple(p.first.data(), range_type{}), p.first.index());
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto position_end(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        using range_type = typename ProxyParticleIterator::range_position_type;
        using sub_tuple_type = std::decay_t<decltype(meta::make_sub_tuple(p.second.data(), range_type{}))>;
        using iterator_type = light_tuple_iterator<sub_tuple_type, ProxyParticleIterator::is_const_qualified>;
        return iterator_type(meta::make_sub_tuple(p.second.data(), range_type{}), p.second.index());
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto inputs_begin(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        using range_type = typename ProxyParticleIterator::range_inputs_type;
        using sub_tuple_type = std::decay_t<decltype(meta::make_sub_tuple(p.first.data(), range_type{}))>;
        using iterator_type = light_tuple_iterator<sub_tuple_type, ProxyParticleIterator::is_const_qualified>;
        return iterator_type(meta::make_sub_tuple(p.first.data(), range_type{}), p.first.index());
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto inputs_end(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        using range_type = typename ProxyParticleIterator::range_inputs_type;
        using sub_tuple_type = std::decay_t<decltype(meta::make_sub_tuple(p.second.data(), range_type{}))>;
        using iterator_type = light_tuple_iterator<sub_tuple_type, ProxyParticleIterator::is_const_qualified>;
        return iterator_type(meta::make_sub_tuple(p.second.data(), range_type{}), p.second.index());
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto outputs_begin(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        using range_type = typename ProxyParticleIterator::range_outputs_type;
        using sub_tuple_type = std::decay_t<decltype(meta::make_sub_tuple(p.first.data(), range_type{}))>;
        using iterator_type = light_tuple_iterator<sub_tuple_type, ProxyParticleIterator::is_const_qualified>;
        return iterator_type(meta::make_sub_tuple(p.first.data(), range_type{}), p.first.index());
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto outputs_end(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        using range_type = typename ProxyParticleIterator::range_outputs_type;
        using sub_tuple_type = std::decay_t<decltype(meta::make_sub_tuple(p.second.data(), range_type{}))>;
        using iterator_type = light_tuple_iterator<sub_tuple_type, ProxyParticleIterator::is_const_qualified>;
        return iterator_type(meta::make_sub_tuple(p.second.data(), range_type{}), p.second.index());
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto variables_begin(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        using range_type = typename ProxyParticleIterator::range_variables_type;
        using sub_tuple_type = std::decay_t<decltype(meta::make_sub_tuple(p.first.data(), range_type{}))>;
        using iterator_type = light_tuple_iterator<sub_tuple_type, ProxyParticleIterator::is_const_qualified>;
        return iterator_type(meta::make_sub_tuple(p.first.data(), range_type{}), p.first.index());
    }

    /**
     * @brief
     *
     * @tparam ProxyParticleIterator
     * @param p
     * @return constexpr auto
     */
    template<typename ProxyParticleIterator>
    constexpr inline auto variables_end(std::pair<ProxyParticleIterator, ProxyParticleIterator> const& p)
    {
        using range_type = typename ProxyParticleIterator::range_variables_type;
        using sub_tuple_type = std::decay_t<decltype(meta::make_sub_tuple(p.second.data(), range_type{}))>;
        using iterator_type = light_tuple_iterator<sub_tuple_type, ProxyParticleIterator::is_const_qualified>;
        return iterator_type(meta::make_sub_tuple(p.second.data(), range_type{}), p.second.index());
    }
}   // namespace scalfmm::container
