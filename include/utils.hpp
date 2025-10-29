#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

// Generate a random vector of integers
inline std::vector<int> generate_random_vector(int n, int min_val = 0, int max_val = 1'000'000) {
    std::vector<int> v(n);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    for (auto &x : v) x = dist(rng);
    return v;
}

// Print vector (for debugging)
inline void print_vector(const std::vector<int>& v, size_t max_elems = 20) {
    size_t n = std::min(v.size(), max_elems);
    for (size_t i = 0; i < n; ++i)
        std::cout << v[i] << " ";
    if (v.size() > max_elems)
        std::cout << "...";
    std::cout << "\n";
}

// Check if vector is sorted
inline bool is_sorted(const std::vector<int>& v) {
    return std::is_sorted(v.begin(), v.end());
}
