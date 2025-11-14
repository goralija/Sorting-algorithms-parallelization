#include <main_template.hpp>
#include <algorithm>
#include <execution>

void std_sort_wrapper(std::vector<int>& vec) {
    // Explicitly use sequential execution policy to prevent any parallelization
    std::sort(std::execution::seq, vec.begin(), vec.end());
}

int main(int argc, char* argv[]) {
    return run_sort("std_sort", "sequential", std_sort_wrapper, argc, argv);
}