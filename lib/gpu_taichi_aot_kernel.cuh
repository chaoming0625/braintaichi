#ifndef TAICHI_KERNEL_GPU_H
#define TAICHI_KERNEL_GPU_H
#include <taichi/cpp/taichi.hpp>
#include <taichi/taichi_cuda.h>
#include <fstream>
#include <ostream>
#include <map>
#include <typeinfo>
#include <typeindex>
#include <memory>


struct TaichiKernel{
    ti::Runtime runtime_;
    std::shared_ptr<ti::Kernel> kernel;
    ti::Kernel kernel_;
    ti::AotModule module_;
    std::map<std::string, std::shared_ptr<ti::Kernel>> kernel_cache;
    std::string current_kernel_path_;

    TaichiKernel(){
        runtime_ = ti::Runtime(TI_ARCH_CUDA);
    }

    void load_from_kernel_aot_path(const char* kernel_aot_path) {
        module_ = runtime_.load_aot_module(kernel_aot_path);
        ti::check_last_error();
        kernel_ = module_.get_kernel("taichi_kernel_gpu");
        ti::check_last_error();
    }

    std::shared_ptr<ti::Kernel>& get_kernel(const std::string& kernel_aot_path) {
        auto it = kernel_cache.find(kernel_aot_path);
        if (it == kernel_cache.end()) {
            load_from_kernel_aot_path(kernel_aot_path.c_str());
            auto kernel_ptr = std::make_shared<ti::Kernel>(std::move(kernel_));
            kernel_cache[kernel_aot_path] = kernel_ptr;
            current_kernel_path_ = kernel_aot_path;
            return kernel_cache[kernel_aot_path];
        }
        current_kernel_path_ = kernel_aot_path;
        return it->second;
    }

    void load(const char* kernel_aot_path) {
        if (current_kernel_path_ == kernel_aot_path) {
            return;
        }
        kernel = get_kernel(kernel_aot_path);
    }

    void clear_args() {
        if (kernel)
            kernel->clear_args();
    }

    void launch(){
        if (kernel) {
            kernel->launch();
            runtime_.wait();
            ti::check_last_error();
        }
    }

    void set_cuda_stream(cudaStream_t stream) {
        ti_set_cuda_stream(stream);
    }
};


extern TaichiKernel *taichi_kernel;

void push_input(const uint32_t type_id, const void* value, uint32_t dim_count, uint32_t elem_count, uint32_t* shape);

void push_output(const uint32_t type_id, const void* value, uint32_t dim_count, uint32_t elem_count, uint32_t* shape);

// void push_args(const uint32_t type_id, const void* value, uint32_t dim_count, uint32_t elem_count, uint32_t* shape);

TiDataType getTiDataTypeFromMap(uint32_t typeIndex);

struct OpaqueStruct {
    uint32_t in_num;
    uint32_t out_num;
    std::vector<uint32_t> type_list;
    std::vector<uint32_t> ndim_list;
    std::vector<uint32_t> shape_list;
    std::vector<uint32_t> size_list;
    std::string kernel_aot_path;
};

OpaqueStruct parseOpaque(const char* opaque, std::size_t opque_len);



template<typename data_type>
ti::NdArray<data_type> createNdArrayFromRawMemory(void* raw_memory, 
                                                  uint32_t dim_count, 
                                                  uint32_t element_count,
                                                  uint32_t dtype,
                                                  uint32_t* shape) {
// #ifdef TI_WITH_LLVM

    // import raw memory
    size_t memory_size = sizeof(data_type) * element_count;

    auto memory = ti_import_cuda_memory(taichi_kernel->runtime_, raw_memory, memory_size);

    // prepare tiNdArray
    TiNdArray tiNdArray;
    tiNdArray.memory = memory;
    tiNdArray.shape.dim_count = dim_count;

    for (int i = 0; i < dim_count; i++) {
        tiNdArray.shape.dims[i] = shape[i];
    }

    tiNdArray.elem_shape.dim_count = 0;
    tiNdArray.elem_type = getTiDataTypeFromMap(dtype);

    auto ti_memory = ti::Memory(taichi_kernel->runtime_, memory, memory_size, false);
    // create NdArray
    auto ndarray = ti::NdArray<data_type>(std::move(ti_memory), tiNdArray);

    return ndarray;
// #else
//     TI_NOT_IMPLEMENTED
// #endif
}

#endif //TAICHI_KERNEL_CPU_H