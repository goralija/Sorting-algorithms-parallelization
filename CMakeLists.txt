cmake_minimum_required(VERSION 3.16)
project(SortingAlgorithmsParallelization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(include)

# Try to find OpenMP, but don’t fail if missing
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "✅ OpenMP found! Enabling OpenMP flags.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "⚠️ OpenMP not found. Compiling without parallel support.")
endif()

# --- Sequential executables ---
file(GLOB SEQ_SRC "sequential_naive/*.cpp")
foreach(src_file ${SEQ_SRC})
    get_filename_component(exec_name ${src_file} NAME_WE)
    add_executable(sequential_naive_${exec_name} ${src_file})
endforeach()

file(GLOB SEQ_SRC "sequential_optimized/*.cpp")
foreach(src_file ${SEQ_SRC})
    get_filename_component(exec_name ${src_file} NAME_WE)
    add_executable(sequential_optimized_${exec_name} ${src_file})
endforeach()

# --- Parallel CPU executables ---
file(GLOB CPU_SRC "parallel_cpu/*.cpp")
foreach(src_file ${CPU_SRC})
    get_filename_component(exec_name ${src_file} NAME_WE)
    add_executable(parallel_cpu_${exec_name} ${src_file})
    if(OpenMP_FOUND)
        target_link_libraries(parallel_cpu_${exec_name} OpenMP::OpenMP_CXX)
    endif()
endforeach()
