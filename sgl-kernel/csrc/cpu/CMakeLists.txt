cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel_flash_mla)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/flash_mla.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_extension_flash_mla.cpp
)

add_compile_options(
    -O3
    -Wno-unknown-pragmas
    -march=native
    -fopenmp
)

Python_add_library(flash_mla_ops MODULE WITH_SOABI ${SOURCES})
target_link_libraries(flash_mla_ops PRIVATE ${TORCH_LIBRARIES})
target_include_directories(flash_mla_ops PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS flash_mla_ops
    LIBRARY DESTINATION sgl_kernel
)
