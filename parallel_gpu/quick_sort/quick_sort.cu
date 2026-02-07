#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdint>

// Definiši uint za IntelliSense
typedef unsigned int uint;

// ============================
// Timer klasa
// ============================

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

public:
    void begin() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    double ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        return duration.count();
    }
};

// ============================
// Utility funkcije
// ============================

std::vector<int> generate_array(size_t n, const std::string& type, unsigned int seed) {
    std::vector<int> arr(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 1000000);

    if (type == "random") {
        for (size_t i = 0; i < n; i++) {
            arr[i] = dis(gen);
        }
    } else if (type == "sorted") {
        for (size_t i = 0; i < n; i++) {
            arr[i] = i;
        }
    } else if (type == "reverse") {
        for (size_t i = 0; i < n; i++) {
            arr[i] = n - i;
        }
    } else if (type == "nearly_sorted") {
        for (size_t i = 0; i < n; i++) {
            arr[i] = i;
        }
        for (size_t i = 0; i < n / 100; i++) {
            int idx1 = dis(gen) % n;
            int idx2 = dis(gen) % n;
            std::swap(arr[idx1], arr[idx2]);
        }
    }
    return arr;
}

void print_vector(const std::vector<int>& vec, size_t max_elements = 20) {
    size_t limit = std::min(max_elements, vec.size());
    for (size_t i = 0; i < limit; i++) {
        std::cout << vec[i];
        if (i < limit - 1) std::cout << " ";
    }
    if (vec.size() > max_elements) {
        std::cout << " ... (and " << (vec.size() - max_elements) << " more)";
    }
    std::cout << std::endl;
}

// ============================
// run_sort() funkcija
// ============================

template <typename SortFunc>
int run_sort(const std::string& algorithm_name, const std::string& mode, SortFunc sort_function, int argc, char* argv[]) {
    // Parse CLI args
    size_t n = 1000000;                // default
    std::string type = "random";       // default
    unsigned int seed = 12345u;        // default
    bool print_array = false;

    if (argc > 1) n = std::stoul(argv[1]);
    if (argc > 2) type = argv[2];
    if (argc > 3) seed = static_cast<unsigned int>(std::stoul(argv[3]));
    if (argc > 4) {
        std::string flag = argv[4];
        if (flag == "--print-array" || flag == "print" || flag == "-p") print_array = true;
    }

    // Generate array
    auto arr = generate_array(n, type, seed);

    // Print initial array
    if (print_array && n <= 1000) {
        std::cout << std::endl << "=== Initial array ===\n";
        print_vector(arr, 200);
    }

    // Run sort and measure time
    Timer t;
    t.begin();
    sort_function(arr);
    t.stop();
    double time_ms = t.ms();

    // Print sorted array
    if (print_array && n <= 1000) {
        std::cout << std::endl << "=== Array after sort ===\n";
        print_vector(arr, 200);
    }

    // Verify that the array is sorted
    bool sorted = std::is_sorted(arr.begin(), arr.end());
    if (!sorted) {
        std::cerr << "Error: Array is NOT sorted after running " << algorithm_name << " (" << mode << ")\n";
        std::ofstream log("errors.log", std::ios::app);
        if (log.is_open()) {
            log << "Algorithm: " << algorithm_name << ", Mode: " << mode
                << ", Size: " << n << ", Type: " << type << " → FAILED (unsorted)\n";
        }
        return 1;
    }

    // Output
    std::cout << "\n=== Results ===\n";
    std::cout << "Algorithm: " << algorithm_name << "\n";
    std::cout << "Mode: " << mode << "\n";
    std::cout << "Array size: " << n << "\n";
    std::cout << "Array type: " << type << "\n";
    std::cout << "Execution time (ms): " << time_ms << "\n";
    std::cout << "Verification: sorted = " << (sorted ? "true" : "false") << "\n";

    // Append results to CSV
    std::ofstream ofs("benchmark.csv", std::ios::app);
    if (ofs.is_open()) {
        ofs << algorithm_name << "," << mode << "," << n << "," << type << "," << time_ms << "\n";
        ofs.close();
    }

    return 0;
}

// ============================
// Device konstante
// ============================

#define BLOCK_SIZE 256
#define MAX_DEPTH 32
#define INSERTION_SORT_THRESHOLD 64
#define BITONIC_SORT_THRESHOLD 512

// ============================
// Device funkcije
// ============================

__device__ inline void swap_kernel(int& a, int& b) {
    int t = a;
    a = b;
    b = t;
}

// ============================
// Kernel: Partition (Hoare scheme)
// ============================

__global__ void partitionKernel(int* arr, int left, int right, int* pivot_pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int pivot = arr[(left + right) / 2];
        int i = left;
        int j = right;
        
        while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            
            if (i <= j) {
                swap_kernel(arr[i], arr[j]);
                i++;
                j--;
            }
        }
        
        *pivot_pos = i;
    }
}

// ============================
// Kernel: Insertion sort za male nizove
// ============================

__global__ void insertionSortKernel(int* arr, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
}

// ============================
// Kernel: Bitonic sort
// ============================

__global__ void bitonicSortKernel(int* arr, int left, int n) {
    int tid = threadIdx.x;
    
    extern __shared__ int shared[];
    
    // Kopiranje u shared memory
    if (tid < n) {
        shared[tid] = arr[left + tid];
    }
    __syncthreads();
    
    // Bitonic sort algoritam
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int idx = tid;
            int xor_val = idx ^ j;
            
            if (xor_val > idx && idx < n && xor_val < n) {
                if ((idx & k) == 0) {
                    if (shared[idx] > shared[xor_val]) {
                        swap_kernel(shared[idx], shared[xor_val]);
                    }
                } else {
                    if (shared[idx] < shared[xor_val]) {
                        swap_kernel(shared[idx], shared[xor_val]);
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Kopiranje rezultata nazad
    if (tid < n) {
        arr[left + tid] = shared[tid];
    }
}

// ============================
// Glavna sort funkcija
// ============================

void gpuQuickSort(std::vector<int>& vec) {
    if (vec.size() <= 1) return;
    
    int n = vec.size();
    int* d_arr = nullptr;
    int* d_pivot_pos = nullptr;
    
    // Alokacija memorije na GPU
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_pivot_pos, sizeof(int));
    
    // Kopiranje niza na GPU
    cudaMemcpy(d_arr, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Stack za iterativnu obradu (umesto rekurzije)
    struct {
        int left, right;
    } stack[MAX_DEPTH];
    
    int stack_top = 0;
    stack[stack_top].left = 0;
    stack[stack_top].right = n - 1;
    stack_top++;
    
    int pivot_pos = 0;
    
    // Obrada stack-a
    while (stack_top > 0) {
        stack_top--;
        int left = stack[stack_top].left;
        int right = stack[stack_top].right;
        
        if (left >= right) continue;
        
        int size = right - left + 1;
        
        // Za male nizove koristi insertion sort
        if (size <= INSERTION_SORT_THRESHOLD) {
            insertionSortKernel<<<1, 1>>>(d_arr + left, size);
            cudaDeviceSynchronize();
            continue;
        }
        
        // Za srednje nizove koristi bitonic sort
        if (size <= BITONIC_SORT_THRESHOLD) {
            int num_threads = 1;
            while (num_threads < size) num_threads *= 2;
            if (num_threads > BLOCK_SIZE) num_threads = BLOCK_SIZE;
            
            bitonicSortKernel<<<1, num_threads, BLOCK_SIZE * sizeof(int)>>>(d_arr, left, size);
            cudaDeviceSynchronize();
            continue;
        }
        
        // Za velike nizove koristi quick sort
        partitionKernel<<<1, 1>>>(d_arr, left, right, d_pivot_pos);
        cudaDeviceSynchronize();
        
        // Preuzmi pivot poziciju sa GPU-a
        cudaMemcpy(&pivot_pos, d_pivot_pos, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Dodaj levi deo na stack
        if (left < pivot_pos - 1 && stack_top < MAX_DEPTH - 1) {
            stack[stack_top].left = left;
            stack[stack_top].right = pivot_pos - 1;
            stack_top++;
        }
        
        // Dodaj desni deo na stack
        if (pivot_pos < right && stack_top < MAX_DEPTH - 1) {
            stack[stack_top].left = pivot_pos;
            stack[stack_top].right = right;
            stack_top++;
        }
    }
    
    // Kopiranje rezultata nazad na CPU
    cudaMemcpy(vec.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Oslobađanje GPU memorije
    cudaFree(d_arr);
    cudaFree(d_pivot_pos);
}

// ============================
// Entry Point - JEDINA main FUNKCIJA
// ============================

int main(int argc, char* argv[]) {
    return run_sort("GPU QuickSort", "gpu_only", gpuQuickSort, argc, argv);
}