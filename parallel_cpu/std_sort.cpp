#include <main_template.hpp>
#include <vector>
#include <parallel/algorithm> // GNU Parallel STL

void std_sort_wrapper(std::vector<int> &vec)
{
    // GNU parallel STL sort (OpenMP inside)
    __gnu_parallel::sort(vec.begin(), vec.end());
}

int main(int argc, char *argv[])
{
    return run_sort("std_sort", "parallel_opt", std_sort_wrapper, argc, argv);
}
