#include "cpu_arm64_taichi_kernel_call.h"
#include "cpu_arm64_taichi_aot_kernel.h"
#include <vector>

namespace brain_taichi {
    void launch_taichi_cpu_arm64_kernel(void **out, const void **in) {
        // The inputs
        const uint32_t in_num = reinterpret_cast<const uint32_t *>(in[0])[0];
        const uint32_t out_num = reinterpret_cast<const uint32_t *>(in[0])[1];

        const uint32_t *type_list = reinterpret_cast<const uint32_t *>(in[1]);
        const uint32_t *dim_count_list = reinterpret_cast<const uint32_t *>(in[2]);
        const uint32_t *elem_count_list = reinterpret_cast<const uint32_t *>(in[3]);
        const uint32_t *shape_list = reinterpret_cast<const uint32_t *>(in[4]);

        const char *kernel_name = reinterpret_cast<const char *>(in[5]);

        taichi_kernel_ARM64->load(kernel_name);

        // restruct shape_list, it's a 2d array and the shape of it is (in_num+out_num, the max of dim_count)
        uint32_t max_dim_count = 0;
        for (int i = 0; i < in_num + out_num; i++) {
            if (dim_count_list[i] > max_dim_count) {
                max_dim_count = dim_count_list[i];
            }
        }

        // Using std::vector to replace the VLA
        std::vector <std::vector<uint32_t>> shape_list_2d(in_num + out_num,
                                                          std::vector<uint32_t>(max_dim_count));

        for (int i = 0; i < in_num + out_num; i++) {
            for (int j = 0; j < max_dim_count; j++) {
                shape_list_2d[i][j] = shape_list[i * max_dim_count + j];
            }
        }


        for (int i = 0; i < in_num; i++) {
            push_input_ARM64(type_list[i],
                             in[6 + i],
                             dim_count_list[i],
                             elem_count_list[i],
                             &shape_list_2d[i][0]);
        }

        for (int i = 0; i < out_num; i++) {
            push_output_ARM64(type_list[in_num + i],
                              out[i],
                              dim_count_list[in_num + i],
                              elem_count_list[in_num + i],
                              &shape_list_2d[in_num + i][0]);
        }

        // launch
        taichi_kernel_ARM64->launch();
        taichi_kernel_ARM64->clear_args();
    }

    void launch_taichi_cpu_arm64_kernel_single_result(void *out, const void **in) {
        // The inputs
        const uint32_t in_num = reinterpret_cast<const uint32_t *>(in[0])[0];
        const uint32_t out_num = reinterpret_cast<const uint32_t *>(in[0])[1];

        const uint32_t *type_list = reinterpret_cast<const uint32_t *>(in[1]);
        const uint32_t *dim_count_list = reinterpret_cast<const uint32_t *>(in[2]);
        const uint32_t *elem_count_list = reinterpret_cast<const uint32_t *>(in[3]);
        const uint32_t *shape_list = reinterpret_cast<const uint32_t *>(in[4]);

        const char *kernel_name = reinterpret_cast<const char *>(in[5]);

        taichi_kernel_ARM64->load(kernel_name);

        // restruct shape_list, it's a 2d array and the shape of it is (in_num+out_num, the max of dim_count)
        uint32_t max_dim_count = 0;
        for (int i = 0; i < in_num + out_num; i++) {
            if (dim_count_list[i] > max_dim_count) {
                max_dim_count = dim_count_list[i];
            }
        }

        // Using std::vector to replace the VLA
        std::vector <std::vector<uint32_t>> shape_list_2d(in_num + out_num, std::vector<uint32_t>(max_dim_count));

        for (int i = 0; i < in_num + out_num; i++) {
            for (int j = 0; j < max_dim_count; j++) {
                shape_list_2d[i][j] = shape_list[i * max_dim_count + j];
            }
        }


        for (int i = 0; i < in_num; i++) {
            push_input_ARM64(type_list[i],
                             in[6 + i],
                             dim_count_list[i],
                             elem_count_list[i],
                             &shape_list_2d[i][0]);
        }

        push_output_ARM64(type_list[in_num],
                          out,
                          dim_count_list[in_num],
                          elem_count_list[in_num],
                          &shape_list_2d[in_num][0]);

        // launch
        taichi_kernel_ARM64->launch();
        taichi_kernel_ARM64->clear_args();
    }
}
