#include "gpu_taichi_kernel_call.cuh"
#include "gpu_taichi_aot_kernel.cuh"
#include <unistd.h>

namespace brain_taichi {
    void launch_taichi_gpu_kernel(cudaStream_t stream, void **buffers,
                                  const char *opaque, std::size_t opaque_len) {
        // cudaDeviceSynchronize();
        // cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
        cudaStreamSynchronize(stream);
        taichi_kernel->set_cuda_stream(stream);

        OpaqueStruct data = parseOpaque(opaque, opaque_len);

        // restruct shape_list, it's a 2d array and the shape of it is (in_num+out_num, the max of dim_count)
        int param_total_num = data.in_num + data.out_num;
        uint32_t shape_list_2d[param_total_num][8];
        for (int i = 0; i < param_total_num; i++) {
            for (int j = 0; j < 8; j++) {
                shape_list_2d[i][j] = data.shape_list[i * 8 + j];
            }
        }

        taichi_kernel->load(data.kernel_aot_path.c_str());

        for (int i = 0; i < data.in_num; i++) {
            push_input(data.type_list[i], buffers[i], data.ndim_list[i], data.size_list[i], shape_list_2d[i]);
        }

        for (int i = 0; i < data.out_num; i++) {
            push_output(data.type_list[i + data.in_num], buffers[i + data.in_num], data.ndim_list[i + data.in_num], data.size_list[i + data.in_num], shape_list_2d[i + data.in_num]);
        }

        // for (int i = 0; i < data.in_num + data.out_num; i++) {
        //     std::cout << "push args: "<< i << " buffer: " << buffers[i] << " ndim: " << data.ndim_list[i] << " size: " << data.size_list[i] << " shape: " << shape_list_2d[i] << std::endl;
        //     push_args(data.type_list[i], buffers[i], data.ndim_list[i], data.size_list[i], shape_list_2d[i]);
        // }

        taichi_kernel->launch();
        taichi_kernel->runtime_.wait();
        taichi_kernel->clear_args();
    }
}
