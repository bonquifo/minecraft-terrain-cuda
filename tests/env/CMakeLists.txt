# Phase 0 probe: proves CMake can drive nvcc + MSVC 14.44 for sm_120 with the
# exactness flags the real build will use.
cmake_minimum_required(VERSION 3.28)
project(env_probe LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)
add_executable(hello_sm120 hello_sm120.cu)
target_compile_options(hello_sm120 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
