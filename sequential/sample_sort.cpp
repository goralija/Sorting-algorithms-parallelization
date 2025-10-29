// merge_sort.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include "../include/timers.hpp" // include your Timer

void merge_sort(std::vector<int>& arr) {
    // Using std::sort as a placeholder
    std::sort(arr.begin(), arr.end());
}

int main(int argc, char* argv[]) {
    int n = 100000; // default size
    if (argc > 1) {
        n = std::stoi(argv[1]);
    }

    // Generate random array
    std::vector<int> arr(n);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 1000000);
    for (int &x : arr) {
        x = dist(rng);
    }

    // Use Timer
    Timer t;
    t.begin();
    merge_sort(arr);
    t.stop();

    std::cout << "Array size: " << n << "\n";
    std::cout << "Execution time (ms): " << t.ms() << "\n";

    // Optional: save results to CSV
    std::ofstream ofs("../data/benchmark.csv", std::ios::app);
    if (ofs.is_open()) {
        ofs << "merge_sort,sequential," << n << "," << t.ms() << ",1\n";
        ofs.close();
    }

    return 0;
}
