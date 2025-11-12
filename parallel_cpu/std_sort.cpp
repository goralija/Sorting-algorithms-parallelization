#include <main_template.hpp>
#include <algorithm>

void std_sort_wrapper(std::vector<int>& vec) {
    std::sort(vec.begin(), vec.end());
}

int main(int argc, char* argv[]) {
    return run_sort("std_sort", "parallel_opt", std_sort_wrapper, argc, argv);
}