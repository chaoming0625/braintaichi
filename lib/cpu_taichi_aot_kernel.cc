#include "cpu_taichi_aot_kernel.h"

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
        memset(const_cast<void*>(value), 0, sizeof(int) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;
    
    case 1:
        memset(const_cast<void*>(value), 0, sizeof(float) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<float>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 2:
        memset(const_cast<void*>(value), 0, sizeof(bool) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<bool>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 3:
        memset(const_cast<void*>(value), 0, sizeof(uint8_t) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint8_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 4:
        memset(const_cast<void*>(value), 0, sizeof(uint16_t) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint16_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 5:
        memset(const_cast<void*>(value), 0, sizeof(uint32_t) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint32_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 6:
        memset(const_cast<void*>(value), 0, sizeof(uint64_t) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<uint64_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 7:
        memset(const_cast<void*>(value), 0, sizeof(int8_t) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int8_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 8:
        memset(const_cast<void*>(value), 0, sizeof(int16_t) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int16_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 9:
        memset(const_cast<void*>(value), 0, sizeof(int64_t) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<int64_t>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 10:
        memset(const_cast<void*>(value), 0, sizeof(float) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<float>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;

    case 11:
        memset(const_cast<void*>(value), 0, sizeof(double) * elem_count);
        taichi_kernel->kernel->push_arg(createNdArrayFromRawMemory<double>(const_cast<void*>(value), dim_count, elem_count, type_id, shape));
        break;
    default:
        break;
    }
}

TiDataType getTiDataTypeFromMap(uint32_t typeIndex) {
    return taichiTypeMap[typeIndex];
}