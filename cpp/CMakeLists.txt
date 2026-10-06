cmake_minimum_required(VERSION 3.15...3.30)
project(seq2seq_c_batcher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python in current environment
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 using Python's installed pybind11 cmake path
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}")

# Build the pybind11 module
pybind11_add_module(seq2seq_c_batcher batch_builder.cpp)

# Target-specific optimizations
if(MSVC)
    target_compile_options(seq2seq_c_batcher PRIVATE
        /O2
        /std:c++17
        /EHsc
        /bigobj
        /arch:AVX2
        /W3
    )
else()
    target_compile_options(seq2seq_c_batcher PRIVATE
        -O3
        -std=c++17
        -march=native
        -Wall
    )
endif()

# Place output binary directly into the parent repository root for easy Python importing
set_target_properties(seq2seq_c_batcher PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
