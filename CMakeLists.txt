cmake_minimum_required(VERSION 3.10)

project(SOME_TIMINGS_KOKKOS)

# Specify the path to the external libraries
set(KOKKOS_PATH "${PROJECT_SOURCE_DIR}/lib/kokkos")
set(MYNVTX_PATH "${PROJECT_SOURCE_DIR}/lib/mynvtx")

# Add Kokkos
add_subdirectory(${KOKKOS_PATH})

#Add mynvtx
add_library(mynvtx INTERFACE)
target_include_directories(mynvtx INTERFACE ${MYNVTX_PATH})

# Create executable
add_executable(my_program main.cpp)

# Link libraries
target_link_libraries(my_program PRIVATE Kokkos::kokkos mynvtx)