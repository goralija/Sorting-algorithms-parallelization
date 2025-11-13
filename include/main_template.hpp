#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <algorithm>
#include "timers.hpp"
#include "utils.hpp"

template <typename SortFunc>
int run_sort(const std::string& algorithm_name, const std::string& mode, SortFunc sort_function, int argc, char* argv[]) {
    // Parse CLI args
    size_t n = 100000000;              // default
    std::string type = "random";    // default
    unsigned int seed = 12345u;     // default deterministic seed
    bool print_array = false;       // default: don't print

#ifdef BENCHMARK_MODE
    // Compile-time benchmark mode: always minimal overhead
    constexpr bool benchmark_mode = true;
#else
    // Normal mode: full verification and I/O
    constexpr bool benchmark_mode = false;
#endif

    if (argc > 1) n = std::stoul(argv[1]);
    if (argc > 2) type = argv[2];
    if (argc > 3) seed = static_cast<unsigned int>(std::stoul(argv[3]));
    if (argc > 4) {
        std::string flag = argv[4];
        if (flag == "--print-array" || flag == "print" || flag == "-p") print_array = true;
    }

    // Generate array based on type and seed
    auto arr = generate_array(n, type, seed);

    // In benchmark mode, minimize I/O and extra operations for VTune profiling
    if (benchmark_mode) {
        // Pure benchmark mode: only sorting, minimal overhead
        sort_function(arr);
        
        // Use the result to prevent dead code elimination (compiler optimizing away the sort)
        // Compute a simple checksum that forces the compiler to keep the sorted array
        volatile long long checksum = 0;
        for (size_t i = 0; i < std::min(n, size_t(100)); ++i) {
            checksum += arr[i];
        }
        
        return 0;
    }

    // Normal mode: full verification and I/O
    // Optionally print the initial array (full print for debugging; you can limit by max elements in print_vector)
    if (print_array) {
        std::cout << std::endl << "=== Initial array (first 200 elems) ===\n";
        print_vector(arr, 200); // prints first up to 200 elems; adjust as needed
    }

    // Run sort and measure time
    Timer t;
    t.begin();
    sort_function(arr);
    t.stop();
    double time_ms = t.ms();

    // After sort and verification, before output -> add:
    if (print_array) {
        std::cout << std::endl << "=== Array after sort (first 200 elems) ===\n";
        print_vector(arr, 200);
    }


    // ✅ Verify that the array is sorted
    bool sorted = std::is_sorted(arr.begin(), arr.end());
    if (!sorted) {
        std::cerr << "Error: Array is NOT sorted after running " << algorithm_name << " (" << mode << ")\n";
        // Optionally: write to a log file for debugging
        std::ofstream log("../data/errors.log", std::ios::app);
        if (log.is_open()) {
            log << "Algorithm: " << algorithm_name << ", Mode: " << mode
                << ", Size: " << n << ", Type: " << type << " → FAILED (unsorted)\n";
        }
        return 1; // Return non-zero to indicate error
    }

    // Output
    std::cout << "Algorithm: " << algorithm_name << "\n";
    std::cout << "Mode: " << mode << "\n";
    std::cout << "Array size: " << n << "\n";
    std::cout << "Array type: " << type << "\n";
    std::cout << "Execution time (ms): " << time_ms << "\n";
    std::cout << "Verification: sorted = " << (sorted ? "true" : "false") << "\n";

    // Append results to CSV
    std::ofstream ofs("../data/benchmark.csv", std::ios::app);
    if (ofs.is_open()) {
        ofs << algorithm_name << "," << mode << "," << n << "," << type << "," << time_ms << "\n";
        ofs.close();
    }

    return 0;
}
