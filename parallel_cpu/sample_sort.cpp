// sample_sort.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include "../include/main_template.hpp"

void sample_sort(std::vector<int>& arr) {
    std::sort(arr.begin(), arr.end());
}

int main(int argc, char* argv[]) {
    return run_sort("sample_sort", "parallel", sample_sort, argc, argv);
}
