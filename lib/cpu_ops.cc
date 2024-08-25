// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

#include "pybind11_kernel_helpers.h"
#include "cpu_taichi_kernel_call.h"
#include "cpu_arm64_taichi_kernel_call.h"

using namespace brain_taichi;

namespace {
    pybind11::dict Registrations() {
      pybind11::dict dict;

      dict["taichi_kernel_aot_call_cpu"] = EncapsulateFunction(launch_taichi_cpu_kernel);
      dict["taichi_kernel_aot_call_cpu_single_result"] = EncapsulateFunction(launch_taichi_cpu_kernel_single_result);
      dict["taichi_kernel_aot_call_cpu_arm64"] = EncapsulateFunction(launch_taichi_cpu_arm64_kernel);
      dict["taichi_kernel_aot_call_cpu_arm64_single_result"] = EncapsulateFunction(launch_taichi_cpu_arm64_kernel_single_result);

      return dict;
    }

    PYBIND11_MODULE(cpu_ops, m) {
        m.def("registrations", &Registrations);
    }

}  // namespace
