add_library(mccuda_core STATIC
  src/rng.cpp
  src/ir.cpp
  src/seed_tables.cpp
  src/volume_eval.cpp
)
target_include_directories(mccuda_core PUBLIC include ${CMAKE_SOURCE_DIR}/third_party)
