//// --------------------------------
//// See LICENCE file at project root
//// File : aligned_allocator.hpp
//// --------------------------------
//#ifndef SCALFMM_MEMORY_ALIGNED_ALLOCATOR_HPP
//#define SCALFMM_MEMORY_ALIGNED_ALLOCATOR_HPP
//
//#include <cstdlib>
//#include <memory>
//#include <new>
//
//// Debug includes
//#include <iomanip>
//#include <iostream>
////
//
//#include <inria/integer_sequence.hpp>
//
// namespace scalfmm::details
//{
//    template<class>
//    struct sfinae_false : std::false_type
//    {
//    };
//
//    template<class T>
//    static auto test_align(int) -> sfinae_false<decltype(align(std::declval<std::size_t>(),
//    std::declval<std::size_t>(),
//                                                               std::declval<T*&>(), std::declval<std::size_t>()))>;
//
//    template<class>
//    static auto test_align(...) -> std::true_type;
//
//    template<class T>
//    struct has_not_align : decltype(test_align<T>(0))
//    {
//    };
//
//    template<class T>
//    using has_not_align_t = typename std::enable_if<has_not_align<T>::value, void>::type;
//}   // namespace scalfmm::details
//
// namespace std
//{
//    /** Shamelessly copied from https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350#c11 */
//    template<typename T = scalfmm::details::has_not_align_t<void>>
//    inline auto align(std::size_t alignment, std::size_t size, T*& ptr, std::size_t& space) -> T*
//    {
//        auto pn = reinterpret_cast<std::uintptr_t>(ptr);
//        std::uintptr_t aligned = (pn + alignment - 1) & -alignment;
//        std::size_t padding = aligned - pn;
//        if(space < size + padding)
//        {
//            return nullptr;
//        }
//        space -= padding;
//        return ptr = reinterpret_cast<void*>(aligned);
//    }
//}   // namespace std
//
///** \file */
///** \brief Multiple arrays allocator
// *
// * Allocates memory to create aligned arrays for types Ts that share one chunk
// * of memory.
// *
// * ~~~~
// * p : pointer to begining of allocated memory
// * i : first array item (eg. int)
// * do: second array item (eg. double)
// *
// * allocated memory block:
// * |--piiiiiiiiiiiiiiii----------------dodododododododododododododododo--|
// *  ^^                 ^^^^^^^^^^^^^^^^
// *   |                 stuffing to align second array
// *   padding to align memory
// * ~~~~
// *
// * Example usage:
// *
// * ~~~~{.cpp}
// * FAlignedAllocator<128, int, double> alloc;
// * std::tuple<int*, double*> pointers = alloc.allocate(5);
// * int*    a = std::get<0>(pointers);
// * double* b = std::get<1>(pointers);
// * // a and b both point to an array of 5 items.
// * // it is garanteed that the memory between the first element of a and the
// * // last element of b belong to the same allocated memory block
// * alloc.deallocate(pointers);
// * ~~~~
// *
// *
// * \tparam Alignment The memory alignment, must be a power of 2
// * \tparam Ts... The types of the arrays
// */
//
// namespace scalfmm::memory
//{
//    template<std::size_t Alignment, typename... Ts>
//    struct aligned_allocator;
//
//    template<std::size_t Alignment, typename... Ts>
//    struct aligned_allocator_impl
//    {
//        /// Types managed by the allocator
//        using value_type_tuple = std::tuple<Ts...>;
//        /// Pointers to to types managed
//        using pointer_tuple = std::tuple<Ts*...>;
//        /// Const pointers
//        using const_pointer_tuple = std::tuple<const Ts*...>;
//
//        static constexpr auto type_index_sequence = inria::make_index_sequence<sizeof...(Ts)>();
//
//        /// Allows using an allocator for an other type with the same alignment
//        template<typename U, typename... Us>
//        struct rebind
//        {
//            using other = aligned_allocator<Alignment, U, Us...>;
//        };
//
//      public:
//        /** Allocate memory for sizeof...(Ts) array of n elements
//         *
//         * All the arrays share the same block of memory (which does not mean they
//         * overlap).
//         *
//         * \param n length of the arrays
//         */
//        auto allocate(std::size_t n) -> pointer_tuple
//        {
//            // allocated storage
//            // + space for difference between original pointer and aligned memory
//            // + minimal size for alignment
//
//            std::size_t initial_offset = std::max(sizeof(void*), Alignment);
//            const std::size_t size = initial_offset + sum(type_array_size<Ts>(n)...);
//
//#ifdef DEBUG_ALLOC
//            {   // DEBUG print
//                auto l = {(std::cerr << type_array_size<Ts>(n) << " ", 0)...};
//                (void)l;
//            }
//#endif
//
//            // Allocate the computed size
//            void* buffer = std::malloc(size);
//            if(buffer == nullptr)
//            {
//                throw std::bad_alloc();
//            }
//
//#ifdef DEBUG_ALLOC
//            {   // DEBUG print
//                std::cerr << "allocated " << size << " bytes at " << buffer << std::endl;
//            }
//#endif
//
//            // holds arguments for the alignment subcalls
//            alignment_arguments align_args{};
//            // Item count
//            align_args.n = n;
//            // Points to the beginning of memory to be aligned (we keep enough space
//            // to store the buffer original address)
//            align_args.ptr = reinterpret_cast<char*>(buffer) + sizeof(void*);
//            // Space available following ptr
//            align_args.ptr_space = size - sizeof(void*);
//
//            // Pointers to each array
//            pointer_tuple ptr_tuple = std::make_tuple(((Ts*)nullptr)...);
//            // Set aligned pointers
//            if(!align_ptr(align_args, ptr_tuple, type_index_sequence))
//            {
//                free(buffer);
//                throw std::bad_alloc();
//            }
//
//            // Get pointer to memory to store offset
//            char** offset_ptr =
//              reinterpret_cast<char**>(reinterpret_cast<char*>(std::get<0>(ptr_tuple)) - sizeof(void*));
//
//            // Store difference between pointer and beginning of allocated storage
//            // just before pointer
//            *offset_ptr = reinterpret_cast<char*>(buffer);
//
//            return ptr_tuple;
//        }
//
//        /** Frees memory that was retrieved by a previous #allocate call
//         *
//         * \param p The pointer tuple returned by the previous #allocate call
//         * \param n The item count per array (unsused)
//         */
//        void deallocate(pointer_tuple p, std::size_t /*n*/)
//        {
//            // Retrieve pointer to beginning of allocated memory
//            char** offset_ptr = reinterpret_cast<char**>(reinterpret_cast<char*>(std::get<0>(p)) - sizeof(void*));
//            void* buffer = *reinterpret_cast<void**>(offset_ptr);
//            // Free memory
//            std::free(buffer);
//        }
//
//      private:
//        /// Holds arguments needed to align each type's pointer
//        struct alignment_arguments
//        {
//            /// Count of elements to insert in the aligned buffer
//            std::size_t n{};
//            /// Pointer to be aligned
//            void* ptr{};
//            /// Bytes available in the allocated buffer starting from ptr
//            std::size_t ptr_space{};
//        };
//
//        /** \brief Aligns the pointers in the pointer tuple
//         *
//         * This method finds the array beginnings in the allocated memory
//         */
//        template<std::size_t... Is>
//        auto align_ptr(alignment_arguments& args, pointer_tuple& ptr_tuple, inria::index_sequence<Is...> /*unused*/)
//        -> bool
//        {
//#ifdef DEBUG_ALLOC
//            {   // DEBUG print
//                std::cerr << std::setw(10) << "type" << std::setw(7) << "sizeof" << std::setw(12) << "origin"
//                          << std::setw(12) << "free space" << std::setw(12) << "array size" << std::setw(12) <<
//                          "offset"
//                          << std::setw(12) << "final" << std::setw(12) << "check" << std::setw(12) << "next"
//                          << std::setw(12) << "rem. space" << std::endl;
//            }
//#endif
//
//            // Initializer-list call order is guaranteed by the standard
//            auto res = {align_ptr_impl<Ts>(args, std::get<Is>(ptr_tuple))...};
//
//            for(bool b: res)
//            {
//                if(!b)
//                {
//                    return false;
//                }
//            }
//            return true;
//        }
//
//        /** \brief Implementation of the alignment algorithm
//         *
//         * \warning This is supposed to be called in the right order for each type
//         *
//         * \param args The arguments used to get and align available memory
//         * \param res_ptr The pointer to set to the aligned memory
//         */
//        template<typename U>
//        auto align_ptr_impl(alignment_arguments& args, U*& res_ptr) -> bool
//        {
//            std::size_t ptr_size = sizeof(U) * args.n;
//
//#ifdef DEBUG_ALLOC
//            char* orig = (char*)args.ptr;
//            {   // DEBUG print
//                // type sizeof(type) size space orig offset to end remaining_space
//                std::cerr << std::setw(10) << typeid(U).name() << std::setw(7) << sizeof(U) << std::setw(12) <<
//                args.ptr
//                          << std::setw(12) << args.ptr_space << std::setw(12) << ptr_size;
//            }
//#endif
//
//            // This call aligns args.ptr and modifies args.ptr_space accordingly
//            if(!std::align(Alignment, ptr_size, args.ptr, args.ptr_space))
//            {
//#ifdef DEBUG_ALLOC
//                {   // DEBUG print
//                    std::cerr << std::endl;
//                }
//#endif
//                return false;
//            }
//
//            // Set the result pointer
//            res_ptr = (U*)args.ptr;
//            // Remove the array size from the available space
//            args.ptr_space -= ptr_size;
//            // move args.ptr past the end of the array for the next align
//            args.ptr = (char*)(res_ptr) + ptr_size;
//
//#ifdef DEBUG_ALLOC
//            {   // DEBUG print
//                std::cerr << std::setw(12) << (char*)res_ptr - orig;
//                std::cerr << std::setw(12) << (void*)res_ptr;
//                std::cerr << std::setw(12) << ((std::size_t)res_ptr % Alignment == 0 ? "ok" : "err.");
//                std::cerr << std::setw(12) << args.ptr;
//                std::cerr << std::setw(12) << args.ptr_space;
//                std::cerr << std::endl;
//            }
//#endif
//            return true;
//        }
//
//        /** Compute the size, in bytes, of the array of n items of type T
//         *
//         * Takes into account the fact that the length must be a multiple of
//         * Alignment
//         *
//         * \return The length of the array rounded up to the
//         */
//        template<typename T>
//        auto type_array_size(std::size_t n) -> std::size_t
//        {
//            // Length of the array
//            std::size_t length = sizeof(T) * n;
//            // Padding to add to get a multiple of Alignment
//            std::size_t stuffing = 0;
//            // Compute stuffing if the length is not already a multiple of Alignment
//            if((length & (Alignment - 1)) != 0)
//            {
//                stuffing = Alignment - (length & (Alignment - 1));
//            }
//            return length + stuffing;
//        }
//
//        /** Variadic sum */
//        template<typename... Args>
//        auto sum(const Args&... args) -> typename std::common_type<Args...>::type
//        {
//            typename std::common_type<Args...>::type res = {};
//            auto l = {(res += args, 0)...};
//            (void)l;
//            return res;
//        }
//
//        // Static assert make doxygen bug
//        static_assert(sizeof...(Ts) > 0, "aligned_allocator needs at least one type.");
//    };
//
//    /** \brief Custom allocator to get aligned memory
//     * \tparam T Type to use memory with.
//     * \tparam Alignement Memory alignment, must be a power of 2.
//     */
//    template<std::size_t Alignment, typename T>
//    struct aligned_allocator<Alignment, T>
//    {
//        /// Type managed by the allocator
//        using value_type = T;
//        /// Pointer to value_type
//        using pointer = T*;
//        /// Const pointer to value_type
//        using const_pointer = const T*;
//
//        /// Allows using an allocator for an other type with the same alignment
//        template<typename U, typename... Us>
//        struct rebind
//        {
//            using other = aligned_allocator<Alignment, U, Us...>;
//        };
//
//        /** \brief Allocate memory for objects of type T
//         *
//         * Allocates more memory than needed to get an aligned buffer :
//         * ~~~~
//         * v-begin   v-pointer to begin              vvv-extra memory
//         * |---------pa-----------------------------e---| < allocated memory
//         * ^-begin    ^-beginning of aligned memory ^-end of aligned memory
//         * ~~~~
//         *
//         * \param n Count of object to allocate memory for
//         */
//        auto allocate(std::size_t n) -> pointer
//        {
//            // allocated storage
//            // + space for difference between original pointer and aligned memory
//            // + minimal size for alignment
//            const std::size_t size = sizeof(T) * n + std::max(sizeof(void*), Alignment);
//
//            // Allocate the computed size
//            void* buffer = std::malloc(size);
//            if(buffer == nullptr)
//            {
//                throw std::bad_alloc();
//            }
//
//            // Pointer to beginning of memory to be aligned (we keep enough space to
//            // store the memory original address)
//            void* ptr = reinterpret_cast<char*>(buffer) + sizeof(void*);
//            // Space available following ptr
//            std::size_t ptr_space = size - sizeof(void*);
//            // Size of the array of objects we want
//            std::size_t ptr_size = sizeof(T) * n;
//            // Get aligned pointer
//            if(std::align(Alignment, ptr_size, ptr, ptr_space) == nullptr)
//            {
//                free(buffer);
//                throw std::bad_alloc();
//            }
//
//            // Get pointer to memory to store offset
//            char** offset_ptr = reinterpret_cast<char**>(reinterpret_cast<char*>(ptr) - sizeof(void*));
//
//            // Store difference between pointer and beginning of allocated storage
//            // just before pointer
//            *offset_ptr = reinterpret_cast<char*>(buffer);
//
//            return (T*)ptr;
//        }
//
//        /** \brief Deallocates previously allocated memory
//         * \param p Pointer to aligned memory
//         * \param n Unused
//         * \warning This does not destroy the objects that may have been constructed.
//         */
//        void deallocate(pointer p, std::size_t /*n*/)
//        {
//            // Retrieve pointer to beginning of allocated memory
//            char** offset_ptr = reinterpret_cast<char**>(reinterpret_cast<char*>(p) - sizeof(void*));
//            void* buffer = *reinterpret_cast<void**>(offset_ptr);
//            // Free memory
//            std::free(buffer);
//        }
//
//        /** \brief Constructs an object of type T without allocating memory
//         * \param p Address where to construct the object
//         * \param args... Arguments to pass to constructor
//         */
//        template<typename... Args>
//        void construct(pointer p, Args... args)
//        {
//            new(p) T(args...);
//        }
//
//        /** \brief Destroy an object of type T without freeing memory
//         * \param p Address of the object
//         */
//        void destroy(pointer p) { p->~T(); }
//
//        /** \brief Equality operator
//         *
//         * If this returns true, then both operands are
//         * equivalent and can free memory allocated by the other.
//         */
//        template<class U, std::size_t AlignU>
//        auto operator==(const aligned_allocator<AlignU, U>& /*other*/) const -> bool
//        {
//            return (sizeof(U) == sizeof(T)) && (AlignU == Alignment);
//        }
//
//        /** Inequality operator */
//        template<class U, std::size_t AlignU>
//        auto operator!=(const aligned_allocator<AlignU, U>& other) const -> bool
//        {
//            return !(*this == other);
//        }
//    };
//
//    template<std::size_t Alignment, typename... Types>
//    struct aligned_allocator : public aligned_allocator_impl<Alignment, Types...>
//    {
//        template<typename U, typename... Us>
//        struct rebind
//        {
//            using other = aligned_allocator<Alignment, U, Us...>;
//        };
//    };
//
//    template<std::size_t Alignment, typename... Types>
//    struct aligned_allocator<Alignment, std::tuple<Types...>> : public aligned_allocator_impl<Alignment, Types...>
//    {
//        template<typename U, typename... Us>
//        struct rebind
//        {
//            using other = aligned_allocator<Alignment, U, Us...>;
//        };
//    };
//}   // End namespace scalfmm::memory
//
//#endif /* ALIGNED_ALLOCATOR_HPP */
