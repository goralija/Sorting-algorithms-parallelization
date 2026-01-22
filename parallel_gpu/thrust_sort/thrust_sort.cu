#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "../../include/main_template.hpp"

// EXPORT/IMPORT zona: Deklarišemo funkcije koje se nalaze u drugim fajlovima
// Linker će ih sam pronaći ako ih kompajliramo zajedno
void merge_sort_gpu(std::vector<int>& host_arr);
void radix_sort_gpu(std::vector<int>& host_arr);

// Naša hibridna verzija (Custom Thrust)
void custom_hybrid_sort(std::vector<int>& h_arr) {
    int n = h_arr.size();

    // 1. Threshold za CPU (ispod 2000 elemenata GPU samo gubi vrijeme)
    if (n < 2048) {
        std::sort(h_arr.begin(), h_arr.end());
        return;
    }

    // 2. Threshold za Merge Sort (između 2k i 1M)
    // Tvoj Merge sa Bitonic bazom je odličan za srednje nizove
    if (n <= 1000000) {
        merge_sort_gpu(h_arr);
    } 
    // 3. Za sve preko milion, koristi Radix
    else {
        radix_sort_gpu(h_arr);
    }
}

// Pravi Thrust za poređenje
void official_thrust_sort(std::vector<int>& h_arr) {
    thrust::device_vector<int> d_vec = h_arr;
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), h_arr.begin());
}

int main(int argc, char* argv[]) {

    cudaFree(0);
    // Pokrećemo našu implementaciju
    run_sort("Custom Hybrid", "Shared-Radix-Merge", custom_hybrid_sort, argc, argv);

    // Pokrećemo zvanični Thrust
    run_sort("Official Thrust", "NVIDIA-Library", official_thrust_sort, argc, argv);

    return 0;
}