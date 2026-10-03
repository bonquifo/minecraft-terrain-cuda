add_executable(mccuda_bench bench_main.cpp)
target_link_libraries(mccuda_bench PRIVATE mccuda_core mccuda_cuda)
target_compile_definitions(mccuda_bench PRIVATE MCCUDA_DATA_DIR="${MCCUDA_DATA_DIR}")
