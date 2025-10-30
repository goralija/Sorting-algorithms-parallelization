// /include/main_template.hpp
#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include "timers.hpp"

// Helper to generate random input data
inline std::vector<int> generate_random_array(size_t n, int min_val = 0, int max_val = 1'000'000) {
    std::vector<int> arr(n);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    for (int &x : arr) x = dist(rng);
    return arr;
}

// Generic runner for sorting algorithms
template <typename SortFunc>
int run_sort(const std::string& algorithm_name, const std::string& mode, SortFunc sort_function, int argc, char* argv[]) {
    size_t n = 100000; // Default size
    if (argc > 1) n = std::stoul(argv[1]);

    auto arr = generate_random_array(n);

    Timer t;
    t.begin();
    sort_function(arr);
    t.stop();

    double time_ms = t.ms();

    std::cout << "Algorithm: " << algorithm_name << "\n";
    std::cout << "Array size: " << n << "\n";
    std::cout << "Execution time (ms): " << time_ms << "\n";

    // Append results to CSV
    std::ofstream ofs("../data/benchmark.csv", std::ios::app);
    if (ofs.is_open()) {
        ofs << algorithm_name << "," << mode << "," << n << "," << time_ms << ",1\n";
        ofs.close();
    }

    return 0;
}
