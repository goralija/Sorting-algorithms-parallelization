#include <main_template.hpp>
#include <algorithm>
#include <execution>

void std_sort_wrapper(std::vector<int> &vec)
{
    // Explicitly use parallel execution policy for parallel version
    std::sort(std::execution::par, vec.begin(), vec.end());
}

int main(int argc, char *argv[])
{
    return run_sort("std_sort", "parallel_opt", std_sort_wrapper, argc, argv);
}