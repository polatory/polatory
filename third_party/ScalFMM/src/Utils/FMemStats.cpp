// See LICENCE file at project root
#include <ScalFMM/Utils/FMemStats.h>

FMemStats FMemStats::controler;

#include <cstdio>
#include <cstdlib>

#ifdef SCALFMM_USE_MEM_STATS
    // Regular scalar new
    void* operator new(std::size_t n) {
        void* const allocated = std::malloc(n + sizeof(size_t));
        if(allocated){
            *(static_cast<size_t*>(allocated)) = n;
            FMemStats::controler.allocate(n);
            return static_cast<unsigned char*>(allocated) + sizeof(size_t);
        }
        throw std::bad_alloc();
        return allocated;
    }

    void* operator new[]( std::size_t n ) {
        void* const allocated = std::malloc(n + sizeof(size_t));
        if(allocated){
            *(static_cast<size_t*>(allocated)) = n;
            FMemStats::controler.allocate(n);
            return static_cast<unsigned char*>(allocated) + sizeof(size_t);
        }
        throw std::bad_alloc();
        return allocated;
    }

    void* operator new  ( std::size_t n, const std::nothrow_t& tag){
        void* const allocated = std::malloc(n + sizeof(size_t));
        if(allocated){
            *(static_cast<size_t*>(allocated)) = n;
            FMemStats::controler.allocate(n);
            return static_cast<unsigned char*>(allocated) + sizeof(size_t);
        }
        return allocated;
    }

    void* operator new[] ( std::size_t n, const std::nothrow_t& tag){
        void* const allocated = std::malloc(n + sizeof(size_t));
        if(allocated){
            *(static_cast<size_t*>(allocated)) = n;
            FMemStats::controler.allocate(n);
            return static_cast<unsigned char*>(allocated) + sizeof(size_t);
        }
        return allocated;
    }

    // Regular scalar delete
    void operator delete(void* p) noexcept{
        if(p){
            FMemStats::controler.deallocate( *(reinterpret_cast<size_t*>(static_cast<unsigned char*>(p) - sizeof(size_t))) );
            std::free(static_cast<unsigned char*>(p) - sizeof(size_t));
        }
    }

    void operator delete[](void* p) noexcept{
        if(p){
            FMemStats::controler.deallocate( *(reinterpret_cast<size_t*>(static_cast<unsigned char*>(p) - sizeof(size_t))) );
            std::free(static_cast<unsigned char*>(p) - sizeof(size_t));
        }
    }

    void operator delete  ( void* p, const std::nothrow_t& /*tag*/) {
        if(p){
            FMemStats::controler.deallocate( *(reinterpret_cast<size_t*>(static_cast<unsigned char*>(p) - sizeof(size_t))) );
            std::free(static_cast<unsigned char*>(p) - sizeof(size_t));
        }
    }

    void operator delete[]( void* p, const std::nothrow_t& /*tag*/) {
        if(p){
            FMemStats::controler.deallocate( *(reinterpret_cast<size_t*>(static_cast<unsigned char*>(p) - sizeof(size_t))) );
            std::free(static_cast<unsigned char*>(p) - sizeof(size_t));
        }
    }

#endif
