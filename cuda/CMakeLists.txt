find_package(Python3 REQUIRED COMPONENTS Interpreter)

# The router evaluators are generated from the locally exported IR (game-derived
# data), so the generated header lives in the build tree and is never committed.
set(MCCUDA_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
add_custom_command(
  OUTPUT ${MCCUDA_GEN_DIR}/mccuda_router.cuh
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MCCUDA_GEN_DIR}
  COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_router.py ${MCCUDA_IR} ${MCCUDA_GEN_DIR}/mccuda_router.cuh
  DEPENDS ${MCCUDA_IR} ${CMAKE_CURRENT_SOURCE_DIR}/codegen/gen_router.py ${CMAKE_CURRENT_SOURCE_DIR}/codegen/ir.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/codegen
  COMMENT "Generating router evaluators from ${MCCUDA_IR}")
add_custom_target(mccuda_codegen DEPENDS ${MCCUDA_GEN_DIR}/mccuda_router.cuh)

add_library(mccuda_cuda STATIC src/generator.cu)
add_dependencies(mccuda_cuda mccuda_codegen)
target_include_directories(mccuda_cuda PUBLIC include PRIVATE ${MCCUDA_GEN_DIR})
target_link_libraries(mccuda_cuda PUBLIC mccuda_core)
set_target_properties(mccuda_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
