add_executable(mccuda_fly fly.cu)
target_link_libraries(mccuda_fly PRIVATE mccuda_core mccuda_cuda)
target_compile_definitions(mccuda_fly PRIVATE MCCUDA_DATA_DIR="${MCCUDA_DATA_DIR}")
set_target_properties(mccuda_fly PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
