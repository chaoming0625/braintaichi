// This header is not specific to our application and you'll probably want something like this
// for any extension you're building. This includes the infrastructure needed to serialize
// descriptors that are used with the "opaque" parameter of the GPU custom call. In our example
// we'll use this parameter to pass the size of our problem.

#ifndef _BRAINTAICHI_KERNEL_HELPERS_CUDA_H_
#define _BRAINTAICHI_KERNEL_HELPERS_CUDA_H_

#include <cstdint>
#include <stdexcept>
#include <cuda_runtime_api.h>


namespace brain_taichi {

    static void ThrowIfError(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }

}  // namespace brain_taichi

#endif
