#pragma once
// run_sort_gpu.hpp
// Same behavior as your CPU run_sort template, but tailored for GPU-based sort functions.
// Usage: provide a SortFunc that accepts std::vector<int>& (same as CPU), and performs
//        device work inside (it must copy data to/from device or accept host vector).
//
// Differences vs run_sort:
// - performs a CUDA warm-up (cudaFree(0))
// - ensures device synchronization after the sort_function returns before stopping the timer
// - writes results to ../data/benchmark_gpu.csv (so GPU runs are tracked separately)
//
// Example:
//   auto gpu_sorter = [](std::vector<int>& a){ merge_sort_gpu(a, /*alg*/2, /*E*/8, /*threads*/256); };
//   run_sort_gpu("Custom Merge Sort", "GPU-Hybrid-Bitonic-Path", gpu_sorter, argc, argv);

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <algorithm>
#include <ctime>
#include "timers.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>

template <typename SortFunc>
int run_sort_gpu(const std::string& algorithm_name, const std::string& mode, SortFunc sort_function, int argc, char* argv[]) {
    // Parse CLI args
    size_t n = 134217728;              // default
    std::string type = "random";       // default
    unsigned int seed = 12345u;        // deterministic default
    bool print_array = false;          // default: don't print

#ifdef BENCHMARK_MODE
    constexpr bool benchmark_mode = true;
#else
    constexpr bool benchmark_mode = false;
#endif

    if (argc > 1) n = std::stoul(argv[1]);
    if (argc > 2) type = argv[2];
    if (argc > 3) seed = static_cast<unsigned int>(std::stoul(argv[3]));
    if (argc > 4) {
        std::string flag = argv[4];
        if (flag == "--print-array" || flag == "print" || flag == "-p") print_array = true;
    }

    // Generate input array
    auto arr = generate_array(n, type, seed);

    // CUDA warm-up (initializes context, reduces first-call variance)
    cudaFree(0);

    if (benchmark_mode) {
        // Pure benchmark mode: only sorting, minimal overhead
        sort_function(arr);

        // Force device completion and prevent the compiler from eliding work
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error (synchronize after sort): " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        volatile long long checksum = 0;
        for (size_t i = 0; i < std::min(n, size_t(100)); ++i) checksum += arr[i];

        (void)checksum;
        return 0;
    }

    // Optional initial print
    if (print_array) {
        std::cout << std::endl << "=== Initial array (first 200 elems) ===\n";
        print_vector(arr, 200);
    }

    // Run sort and measure (ensure device synchronization)
    Timer t;
    t.begin();
    sort_function(arr);
    // ensure GPU finished before stopping timer
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        // continue to try to read results/verify, but report error
    }
    t.stop();
    double time_ms = t.ms();

    if (print_array) {
        std::cout << std::endl << "=== Array after sort (first 200 elems) ===\n";
        print_vector(arr, 200);
    }

    // Verify sorted
    bool sorted = std::is_sorted(arr.begin(), arr.end());
    if (!sorted) {
        std::cerr << "Error: Array is NOT sorted after running " << algorithm_name << " (" << mode << ")\n";
        std::ofstream log("../data/errors_gpu.log", std::ios::app);
        if (log.is_open()) {
            log << "Algorithm: " << algorithm_name << ", Mode: " << mode
                << ", Size: " << n << ", Type: " << type << " → FAILED (unsorted)\n";
            log.close();
        }
        return 1;
    }

    // Output summary
    std::cout << "Algorithm: " << algorithm_name << "\n";
    std::cout << "Mode: " << mode << "\n";
    std::cout << "Array size: " << n << "\n";
    std::cout << "Array type: " << type << "\n";
    std::cout << "Execution time (ms): " << time_ms << "\n";
    std::cout << "Verification: sorted = " << (sorted ? "true" : "false") << "\n";

    // Append results to GPU CSV
    std::ofstream ofs("../data/benchmark_gpu.csv", std::ios::app);
    if (ofs.is_open()) {
        // timestamp, device name, algorithm, mode, n, type, time_ms
        int dev = 0;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            std::time_t ts = std::time(nullptr);
            char timestr[64];
    #ifdef _WIN32
            struct tm tm_buf; localtime_s(&tm_buf, &ts); std::strftime(timestr, sizeof(timestr), "%Y-%m-%d %H:%M:%S", &tm_buf);
    #else
            struct tm tm_buf; localtime_r(&ts, &tm_buf); std::strftime(timestr, sizeof(timestr), "%Y-%m-%d %H:%M:%S", &tm_buf);
    #endif
            ofs << timestr << "," << "\"" << prop.name << "\"," << algorithm_name << "," << mode << "," << n << "," << type << "," << time_ms << "\n";
        } else {
            ofs << algorithm_name << "," << mode << "," << n << "," << type << "," << time_ms << "\n";
        }
        ofs.close();
    } else {
        std::cerr << "Warning: could not open ../data/benchmark_gpu.csv for writing\n";
    }

    return 0;
}