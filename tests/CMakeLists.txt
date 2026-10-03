add_executable(mccuda_tests
  main.cpp
  oracle_data.cpp
  test_l0.cpp
  test_chunks.cpp
  cuda_impls.cpp
)
target_link_libraries(mccuda_tests PRIVATE mccuda_core mccuda_cuda)
target_include_directories(mccuda_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(mccuda_tests PRIVATE MCCUDA_DATA_DIR="${MCCUDA_DATA_DIR}")

enable_testing()
add_test(NAME bitexact_quick COMMAND mccuda_tests --set quick)
add_test(NAME bitexact_full COMMAND mccuda_tests --set full --only chunks)
