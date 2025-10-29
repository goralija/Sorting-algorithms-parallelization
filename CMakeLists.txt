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
file(GLOB SEQ_SRC "sequential/*.cpp")
foreach(src_file ${SEQ_SRC})
    get_filename_component(exec_name ${src_file} NAME_WE)
    add_executable(sequential_${exec_name} ${src_file})
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
