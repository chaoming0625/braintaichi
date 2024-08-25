#include "gpu_taichi_aot_kernel.cuh"
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>

TaichiKernel *taichi_kernel = new TaichiKernel();

std::map<uint32_t, TiDataType> taichiTypeMap = {
    {0, TI_DATA_TYPE_I32},
    {1, TI_DATA_TYPE_F32},
    {2, TI_DATA_TYPE_U1},
    {3, TI_DATA_TYPE_U8},
    {4, TI_DATA_TYPE_U16},
    {5, TI_DATA_TYPE_U32},
    {6, TI_DATA_TYPE_U64},
    {7, TI_DATA_TYPE_I8},
    {8, TI_DATA_TYPE_I16},
    {9, TI_DATA_TYPE_I64},
    {10, TI_DATA_TYPE_F16},
    {11, TI_DATA_TYPE_F64}
};

void push_input(const uint32_t type_id, const void* value, uint32_t dim_count, uint32_t elem_count, uint32_t* shape) {
    switch (type_id)
    {
    case 0:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 1:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<float>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 2:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<bool>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 3:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint8_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 4:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint16_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 5:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint32_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 6:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint64_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 7:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int8_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 8:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int16_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 9:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int64_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 10:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<float>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 11:
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<double>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    default:
        break;
    }
}

void push_output(const uint32_t type_id, const void* value, uint32_t dim_count, uint32_t elem_count, uint32_t* shape) {
    switch (type_id)
    {
    case 0:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(int));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 1:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(float));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<float>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 2:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(bool));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<bool>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 3:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(uint8_t));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint8_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 4:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(uint16_t));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint16_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 5:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(uint32_t));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint32_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 6:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(uint64_t));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint64_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 7:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(int8_t));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int8_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 8:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(int16_t));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int16_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 9:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(int64_t));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int64_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 10:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(float));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<float>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 11:
        cudaMemset(const_cast<void*>(value), 0, elem_count * sizeof(double));
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<double>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    default:
        break;
    }
}


TiDataType getTiDataTypeFromMap(uint32_t typeIndex) {
    return taichiTypeMap[typeIndex];
}

OpaqueStruct parseOpaque(const char* opaque, std::size_t opque_len) {
    std::string input(opaque, opque_len);
    std::stringstream ss(input);
    std::string segment;

    OpaqueStruct data;

    // parse in_num and out_num
    if (std::getline(ss, segment, ';')) {
        std::stringstream nums(segment);
        std::string num;
        std::getline(nums, num, ',');
        data.in_num = std::stoul(num);
        std::getline(nums, num, ',');
        data.out_num = std::stoul(num);
    }

    // Helper function to parse a list of uint32_t
    auto parseList = [&ss](std::vector<uint32_t>& vec) {
        std::string segment;
        if (std::getline(ss, segment, ';')) {
            std::stringstream listStream(segment);
            std::string item;
            while (std::getline(listStream, item, ',')) {
                vec.push_back(std::stoul(item));
            }
        }
    };

    // Parse remaining lists
    parseList(data.type_list);
    parseList(data.ndim_list);
    parseList(data.size_list);
    parseList(data.shape_list);

    // Parse kernel_aot_path
    std::getline(ss, data.kernel_aot_path, ';');

    return data;
}
