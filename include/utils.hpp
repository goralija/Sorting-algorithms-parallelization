#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

// === Array Generators ===

// Fully random array
inline std::vector<int> generate_random_array(size_t n, int min_val = 0, int max_val = 1'000'000) {
    std::vector<int> arr(n);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    for (int &x : arr) x = dist(rng);
    return arr;
}

// Sorted ascending
inline std::vector<int> generate_sorted_array(size_t n) {
    std::vector<int> arr(n);
    for (size_t i = 0; i < n; ++i) arr[i] = static_cast<int>(i);
    return arr;
}

// Reversed (descending)
inline std::vector<int> generate_reversed_array(size_t n) {
    std::vector<int> arr(n);
    for (size_t i = 0; i < n; ++i) arr[i] = static_cast<int>(n - i);
    return arr;
}

// Nearly sorted (mostly sorted with a few random swaps)
inline std::vector<int> generate_nearly_sorted_array(size_t n, double swap_fraction = 0.01) {
    auto arr = generate_sorted_array(n);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    size_t num_swaps = static_cast<size_t>(n * swap_fraction);
    for (size_t i = 0; i < num_swaps; ++i) {
        size_t a = dist(rng), b = dist(rng);
        std::swap(arr[a], arr[b]);
    }
    return arr;
}

// Few unique values
inline std::vector<int> generate_few_unique_array(size_t n, int unique_values = 10) {
    std::vector<int> arr(n);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, unique_values - 1);
    for (int &x : arr) x = dist(rng);
    return arr;
}

// === Unified Array Generator Selector ===
inline std::vector<int> generate_array(size_t n, const std::string& type) {
    if (type == "sorted") return generate_sorted_array(n);
    if (type == "reversed") return generate_reversed_array(n);
    if (type == "nearly_sorted") return generate_nearly_sorted_array(n);
    if (type == "few_unique") return generate_few_unique_array(n);
    return generate_random_array(n); // default
}

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
