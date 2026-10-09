cmake_minimum_required(VERSION 3.28)
project(mccuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Bit-exactness flags (apply to every target) ------------------------------
# Java float/double arithmetic is IEEE-754 with round-to-nearest and no fused
# operations. Host: MSVC /fp:precise does not contract a*b+c into FMA (VS 2022+);
# never /fp:fast or /fp:contract. Device: no FMA contraction, no flush-to-zero,
# IEEE division and square root; never --use_fast_math.
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fp:precise> $<$<COMPILE_LANGUAGE:CXX>:/W3>
                      $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
                      $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-fmad=false> $<$<COMPILE_LANGUAGE:CUDA>:-ftz=false>
                    $<$<COMPILE_LANGUAGE:CUDA>:-prec-div=true> $<$<COMPILE_LANGUAGE:CUDA>:-prec-sqrt=true>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:precise> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
                    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set(MCCUDA_DATA_DIR "${CMAKE_SOURCE_DIR}/data" CACHE PATH "Oracle/game data directory (gitignored)")
set(MCCUDA_MC_VERSION "26.3")
set(MCCUDA_IR "${MCCUDA_DATA_DIR}/gamedata/${MCCUDA_MC_VERSION}/overworld_ir.json")
set(MCCUDA_TERRAIN "${MCCUDA_DATA_DIR}/gamedata/${MCCUDA_MC_VERSION}/overworld_terrain.json")

add_subdirectory(cpu)
add_subdirectory(cuda)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(viz)
