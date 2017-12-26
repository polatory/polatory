// See LICENCE file at project root
#ifndef FMEMSTATS_H
#define FMEMSTATS_H

#include "FGlobal.hpp"

#include <cstdlib>
#include <cstring>
#include <cstdio>

/** Memstat has to be enabled in the cmake,
  * then it will use the know each allocate and deallocate
  * and give simple stats like max, total used, current used
  */

#ifdef SCALFMM_USE_MEM_STATS
#include <new>
#include <stdexcept>
#warning You are using mem stats
void* operator new(std::size_t n);
void* operator new[]( std::size_t n );

void* operator new  ( std::size_t n, const std::nothrow_t& tag);
void* operator new[]( std::size_t n, const std::nothrow_t& tag);

void operator delete(void* p) noexcept;
void operator delete[](void* p) noexcept;
void operator delete  ( void* ptr, const std::nothrow_t& tag);
void operator delete[]( void* ptr, const std::nothrow_t& tag);
#endif

/** Give the memory allocation details
  *
  */
class FMemStats {
private:
    unsigned long long numberOfAllocations;
    unsigned long long maxAllocated;
    unsigned long long totalAllocated;
    std::size_t currentAllocated;

    FMemStats()
        : numberOfAllocations(0), maxAllocated(0), totalAllocated(0), currentAllocated(0) {
    }

#ifdef SCALFMM_USE_MEM_STATS
    ~FMemStats(){
        plotState();
    }
#endif

    void plotState() const {
#ifdef SCALFMM_USE_MEM_STATS
        printf("[SCALFMM-MEMSTAT] Total number of allocations %lld \n", numberOfAllocations);
        printf("[SCALFMM-MEMSTAT] Memory used at the end %lu Bytes (%f MB)\n", FMemStats::controler.getCurrentAllocated(), FMemStats::controler.getCurrentAllocatedMB());
        printf("[SCALFMM-MEMSTAT] Max memory used %lld Bytes (%f MB)\n", FMemStats::controler.getMaxAllocated(), FMemStats::controler.getMaxAllocatedMB());
        printf("[SCALFMM-MEMSTAT] Total memory used %lld Bytes (%f MB)\n", FMemStats::controler.getTotalAllocated(), FMemStats::controler.getTotalAllocatedMB());
#else
        printf("[SCALFMM-MEMSTAT] unused\n");
#endif
    }

    void allocate(const std::size_t size){
        numberOfAllocations += 1;
        currentAllocated += size;
        totalAllocated   += size;

        if(maxAllocated < currentAllocated){
            maxAllocated = currentAllocated;
        }
    }

    void deallocate(const std::size_t size){
        currentAllocated -= size;
    }

#ifdef SCALFMM_USE_MEM_STATS
    friend void* operator new(std::size_t n);
    friend void* operator new[]( std::size_t n );
    friend void* operator new  ( std::size_t n, const std::nothrow_t& tag);
    friend void* operator new[]( std::size_t n, const std::nothrow_t& tag);
    friend void operator delete(void* p) noexcept;
    friend void operator delete[](void* p) noexcept;
    friend void operator delete  ( void* ptr, const std::nothrow_t& tag);
    friend void operator delete[]( void* ptr, const std::nothrow_t& tag);
#endif

public:
    /** Singleton */
    static FMemStats controler;

    /** return the max that has been allocated */
    unsigned long long getNumberOfAllocations() const{
        return numberOfAllocations;
    }

    /** return the max that has been allocated */
    unsigned long long getMaxAllocated() const{
        return maxAllocated;
    }

    /** return the total memory allocated during the running */
    unsigned long long getTotalAllocated() const{
        return totalAllocated;
    }

    /** return the current size allcoated */
    std::size_t getCurrentAllocated() const{
        return currentAllocated;
    }

    /** get the max in MB */
    float getMaxAllocatedMB() const{
        return float(getMaxAllocated()) / 1024 / 1024;
    }

    /** get the total in MB */
    float getTotalAllocatedMB() const{
        return float(getTotalAllocated()) / 1024 / 1024;
    }

    /** get the current in MB */
    float getCurrentAllocatedMB() const{
        return float(getCurrentAllocated()) / 1024 / 1024;
    }

    /** To know if mem stat has been enabled */
    bool isUsed() const {
#ifdef SCALFMM_USE_MEM_STATS
        return true;
#else
        return false;
#endif
    }
};


#endif // FMEMSTATS_H
