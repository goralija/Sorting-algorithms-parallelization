#include <iostream>
#include <vector>
#include <algorithm>
#include "main_template.hpp"
#include <omp.h>

// std::sort wrapper koji koristi obični single-threaded sort bez AVX i bez paralelizma
void std_sort_wrapper(std::vector<int> &vec)
{
    std::sort(vec.begin(), vec.end());
}

int main(int argc, char *argv[])
{
    omp_set_num_threads(1);
    return run_sort("std_sort", "sequential", std_sort_wrapper, argc, argv);
}
