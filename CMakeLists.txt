cmake_minimum_required(VERSION 3.16)
project(SortingAlgorithmsParallelization LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Benchmark mode option ---
option(BENCHMARK_MODE "Enable benchmark mode (minimal overhead, no verification)" OFF)
if(BENCHMARK_MODE)
    add_compile_definitions(BENCHMARK_MODE=1)
    message(STATUS "✅ BENCHMARK_MODE enabled (compile-time)")
else()
    message(STATUS "ℹ️ Normal mode (full verification and I/O)")
endif()

# --- Optimizacija ---
# -O3 za agresivne optimizacije
# -march=native uključi najjači SIMD za tvoj CPU (AVX2, AVX-512 ako postoji)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

include_directories(include)

# --- OpenMP podrška ---
find_package(OpenMP)
if (OpenMP_FOUND)
    message(STATUS "✅ OpenMP found! Enabling OpenMP and SIMD support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "⚠️ OpenMP not found. Compiling without parallel support.")
endif()

# Ako koristiš Clang na macOS-u, treba dodatno:
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND APPLE)
    message(STATUS "ℹ️ macOS + Clang detected → adding OpenMP runtime flags.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
endif()

# --- Sequential executables (naivni) ---
file(GLOB SEQ_NAIVE_SRC "sequential_naive/*.cpp")
foreach(src_file ${SEQ_NAIVE_SRC})
    get_filename_component(exec_name ${src_file} NAME_WE)
    add_executable(sequential_naive_${exec_name} ${src_file})
    # Explicitly disable OpenMP for sequential builds to prevent any parallel optimization
    target_compile_options(sequential_naive_${exec_name} PRIVATE -fno-openmp)
endforeach()

# --- Sequential optimized executables ---
file(GLOB SEQ_OPT_SRC "sequential_optimized/*.cpp")
foreach(src_file ${SEQ_OPT_SRC})
    get_filename_component(exec_name ${src_file} NAME_WE)
    add_executable(sequential_optimized_${exec_name} ${src_file})
    # Explicitly disable OpenMP for sequential builds to prevent any parallel optimization
    target_compile_options(sequential_optimized_${exec_name} PRIVATE -fno-openmp)
endforeach()

# --- Parallel CPU executables (GNU Parallel STL) ---
# --- Parallel CPU executables (OpenMP + SIMD) ---
file(GLOB CPU_SRC "parallel_cpu/*.cpp")

foreach(src_file ${CPU_SRC})
    get_filename_component(exec_name ${src_file} NAME_WE)

    # Posebno izuzmi parallel_std_sort.cpp
    if(NOT exec_name STREQUAL "std_sort")
        add_executable(parallel_cpu_${exec_name} ${src_file})

        if (OpenMP_FOUND)
            target_compile_options(parallel_cpu_${exec_name} PRIVATE ${OpenMP_CXX_FLAGS})
            target_link_libraries(parallel_cpu_${exec_name} OpenMP::OpenMP_CXX)
        endif()

        # eksplicitno SIMD
        target_compile_options(parallel_cpu_${exec_name} PRIVATE -mavx2)
    endif()
endforeach()

# --- Parallel std::sort executable ---
add_executable(parallel_cpu_std_sort parallel_cpu/std_sort.cpp)

# OpenMP podrška za GNU Parallel STL
if (OpenMP_FOUND)
    target_compile_options(parallel_cpu_std_sort PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(parallel_cpu_std_sort OpenMP::OpenMP_CXX)
endif()

# SIMD opcionalno
target_compile_options(parallel_cpu_std_sort PRIVATE -mavx2)


