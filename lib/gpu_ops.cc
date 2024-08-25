// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actual implementation of the
// custom call can be found in kernels.cc.cu.

#include "pybind11_kernel_helpers.h"
#include "gpu_taichi_kernel_call.cuh"

using namespace brain_taichi;

namespace {
    pybind11::dict Registrations() {
        pybind11::dict dict;

        dict["taichi_kernel_aot_call_gpu"] = EncapsulateFunction(launch_taichi_gpu_kernel);

        return dict;
    }

    PYBIND11_MODULE(gpu_ops, m)
    {
        m.def("registrations", &Registrations);
    }
} // namespace
