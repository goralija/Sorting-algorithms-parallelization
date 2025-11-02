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
    size_t n = 100000; // default
    std::string type = "random";
    if (argc > 1) n = std::stoul(argv[1]);
    if (argc > 2) type = argv[2];

    // Generate array based on type
    auto arr = generate_array(n, type);

    // Run sort and measure time
    Timer t;
    t.begin();
    sort_function(arr);
    t.stop();
    double time_ms = t.ms();

    // Output
    std::cout << "Algorithm: " << algorithm_name << "\n";
    std::cout << "Mode: " << mode << "\n";
    std::cout << "Array size: " << n << "\n";
    std::cout << "Array type: " << type << "\n";
    std::cout << "Execution time (ms): " << time_ms << "\n";

    // Append results to CSV
    std::ofstream ofs("../data/benchmark.csv", std::ios::app);
    if (ofs.is_open()) {
        ofs << algorithm_name << "," << mode << "," << n << "," << type << "," << time_ms << "\n";
        ofs.close();
    }

    return 0;
}
