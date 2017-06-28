#ifndef FSTARPUREDUX_HPP
#define FSTARPUREDUX_HPP

#include "FStarPUUtils.hpp"

namespace FStarPUReduxCpu {
template <class DataType>
inline void InitData(void *buffers[], void */*args*/){
    DataType* data = (DataType*)STARPU_VARIABLE_GET_PTR(buffers[0]);
    const size_t length = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0])/sizeof(DataType);

    for(size_t idx = 0 ; idx < length ; ++idx){
        data[idx] = DataType();
    }
}

template <class DataType>
inline void ReduceData(void *buffers[], void */*args*/){
    DataType* dest = (DataType*)STARPU_VARIABLE_GET_PTR(buffers[0]);
    const DataType* src = (DataType*)STARPU_VARIABLE_GET_PTR(buffers[1]);
    const size_t length = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0])/sizeof(DataType);

    for(size_t idx = 0 ; idx < length ; ++idx){
        dest[idx] += src[idx];
    }
}

template <class DataType>
inline void EmptyCodelet(void */*buffers*/[], void */*args*/){
}

}


#endif // FSTARPUREDUX_HPP

