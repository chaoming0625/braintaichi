#ifndef _TAICHI_KERNEL_CALL_GPU_H
#define _TAICHI_KERNEL_CALL_GPU_H

#include "kernel_helpers_gpu.cuh"

namespace brain_taichi {
    void launch_taichi_gpu_kernel(cudaStream_t stream, void **buffers,
                                  const char *opaque, std::size_t opaque_len);
}


#endif //_TAICHI_KERNEL_CALL_GPU_H