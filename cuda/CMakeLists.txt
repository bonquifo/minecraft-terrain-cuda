add_library(mccuda_cuda STATIC src/device_info.cu)
target_include_directories(mccuda_cuda PUBLIC include)
target_link_libraries(mccuda_cuda PUBLIC mccuda_core)
