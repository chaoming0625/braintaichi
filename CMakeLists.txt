set(CMAKE_THREAD_LIBS_INIT "-lpthread")
cmake_minimum_required(VERSION 3.12...3.18)
project(braintaichi LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
set(PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR})
set(PYTHON_LIBRARY ${PYTHON_LIBRARY})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TAICHI_AOT_APP_NAME TaichiKernel)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


if (BRAINPY_CUDA)
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcudart")
endif ()

if(WIN32)
   find_package(Python REQUIRED COMPONENTS Interpreter Development)
else()
    FIND_PACKAGE(PythonInterp 3)
    FIND_PACKAGE(PythonLibs 3)
endif()
find_package(pybind11 REQUIRED)
find_package(Taichi REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${TAICHI_C_API_INSTALL_DIR}/include
    ./pybind11/include
)


include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)


if (BRAINPY_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    pybind11_add_module(
            gpu_ops
            ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc
            ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_taichi_aot_kernel.cu
            ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_taichi_kernel_call.cu
    )
    target_sources(gpu_ops PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_ops.cc
            ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_taichi_aot_kernel.cu
            ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_taichi_kernel_call.cu
    )
    target_link_libraries(gpu_ops PRIVATE Taichi::Runtime)
    # target_link_libraries(gpu_ops PRIVATE ${PYTHON_LIBRARIES})
    install(TARGETS gpu_ops DESTINATION braintaichi)
else()
    message(STATUS "Building without CUDA")
endif ()


pybind11_add_module(
        cpu_ops
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_taichi_aot_kernel.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_taichi_kernel_call.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_arm64_taichi_aot_kernel.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_arm64_taichi_kernel_call.cc
        
)
target_sources(cpu_ops PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_ops.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_taichi_aot_kernel.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_taichi_kernel_call.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_arm64_taichi_aot_kernel.cc
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_arm64_taichi_kernel_call.cc
)
target_link_libraries(cpu_ops PRIVATE Taichi::Runtime)
if(WIN32)
    target_link_libraries(cpu_ops PRIVATE ${PYTHON_LIBRARIES})
endif()
install(TARGETS cpu_ops DESTINATION braintaichi)
