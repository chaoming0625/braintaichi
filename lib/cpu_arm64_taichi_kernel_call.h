#ifndef _TAICHI_KERNEL_CALL_CPU_ARM64_H
#define _TAICHI_KERNEL_CALL_CPU_ARM64_H

#include <cstdint>
#include <cstring>
#include <cmath>

namespace brain_taichi {
    void launch_taichi_cpu_arm64_kernel(void **out, const void **in);
    void launch_taichi_cpu_arm64_kernel_single_result(void *out, const void **in);
}

#endif //_TAICHI_KERNEL_CALL_CPU_ARM64_H