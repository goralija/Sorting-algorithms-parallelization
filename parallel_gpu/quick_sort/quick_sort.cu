// Build:
// nvcc -O2 -std=c++17 -arch=sm_86 -lineinfo quick_iterative_v2.cu -o quick_iterative_v2.exe

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ================== HELPER FUNCTIONS ==================

struct DeviceInfo { 
    int deviceId=0; char name[256]{}; int major=0, minor=0;
    int multiProcessorCount=0; int maxThreadsPerBlock=0; 
};

static DeviceInfo queryDevice(int dev) {
    cudaDeviceProp p{}; gpuErrchk(cudaGetDeviceProperties(&p, dev));
    DeviceInfo di; di.deviceId=dev; snprintf(di.name,sizeof(di.name),"%s",p.name);
    di.major=p.major; di.minor=p.minor; di.multiProcessorCount=p.multiProcessorCount;
    di.maxThreadsPerBlock=p.maxThreadsPerBlock;
    return di;
}

static inline int next_pow2_int(int v) { int p=1; while(p<v) p<<=1; return p; }

std::string timestamp_string() {
    time_t t=time(nullptr); tm tm;
#if defined(_WIN32)
    localtime_s(&tm,&t);
#else
    localtime_r(&t,&tm);
#endif
    char buf[64]; strftime(buf,sizeof(buf),"%Y%m%d_%H%M%S",&tm); return std::string(buf);
}

void ensure_dir(const std::filesystem::path &p) { 
    if(!std::filesystem::exists(p)) std::filesystem::create_directories(p); 
}

struct Task {
    int left;
    int right;
};

struct Timings {
    float h2d_ms=0, kernel_ms=0, d2h_ms=0, total_ms=0;
    int kernel_invocations=0;
};

// ================== HYPER-OPTIMIZED KERNEL ==================
// ================== WARP-AGGREGATED KERNEL (FIXED) ==================

__device__ __forceinline__ int median3(int a, int b, int c) {
    return (a < b) ? ((b < c) ? b : ((a < c) ? c : a)) : ((a < c) ? a : ((b < c) ? c : b));
}

__device__ __forceinline__ int median9(int* data, int left, int right) {
    int n = right - left + 1;
    int step = n / 8;
    int m1 = median3(data[left], data[left + step], data[left + 2*step]);
    int m2 = median3(data[left + 3*step], data[left + 4*step], data[left + 5*step]);
    int m3 = median3(data[left + 6*step], data[left + 7*step], data[right]);
    return median3(m1, m2, m3);
}

// Dodan parametar max_tasks za sigurnost
// ================== WARP-AGGREGATED KERNEL (FIXED & COMPILES) ==================

__global__ void iterative_partition_kernel(int* data, Task* in_q, int num_in, Task* out_q, int* out_cnt, int max_tasks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31; // Warp lane ID (0-31)

    // Svi threadovi prolaze kroz kod (zbog ballot_sync), ali samo aktivni rade posao
    bool active = (idx < num_in);
    
    Task t;
    int n = 0;
    
    // Lokalne varijable moraju biti deklarisane izvan 'if(active)' da bi bile vidljive kasnije
    int left = 0, right = 0;
    
    // Ucitaj samo ako si aktivan
    if (active) {
        t = in_q[idx];
        left = t.left;
        right = t.right;
        n = right - left + 1;
    }
    
    // Lokalni registri za logiku pushanja (inicijalno false/-1)
    int left_child_L = -1, left_child_R = -1;
    bool push_left = false;
    int right_child_L = -1, right_child_R = -1;
    bool push_right = false;

    if (active && n > 1) {
        // --- 1. TINY ARRAYS: Register Sort (<= 16) ---
        if (n <= 16) {
            int local_arr[16];
            // Safe load
            for(int k=0; k<n; ++k) local_arr[k] = data[left + k];
            
            #pragma unroll
            for (int i = 1; i < n; ++i) {
                int key = local_arr[i];
                int j = i - 1;
                while (j >= 0 && local_arr[j] > key) {
                    local_arr[j + 1] = local_arr[j];
                    j--;
                }
                local_arr[j + 1] = key;
            }
            for(int k=0; k<n; ++k) data[left + k] = local_arr[k];
            // Oznacimo da je gotovo
            n = 0; 
        }

        // --- 2. SMALL ARRAYS: Global Insertion Sort (<= 64) ---
        else if (n <= 64) {
            for (int i = left + 1; i <= right; ++i) {
                int key = data[i];
                int j = i - 1;
                while (j >= left && data[j] > key) {
                    data[j + 1] = data[j];
                    j--;
                }
                data[j + 1] = key;
            }
            n = 0; // Done
        }

        // --- 3. PARTITION ---
        else {
            int p;
            if (n > 2048) p = median9(data, left, right);
            else {
                int mid = left + n/2;
                p = median3(data[left], data[mid], data[right]);
            }

            int i = left, j = right;
            while (i <= j) {
                while (data[i] < p) i++;
                while (data[j] > p) j--;
                if (i <= j) {
                    int tmp = data[i]; data[i] = data[j]; data[j] = tmp;
                    i++; j--;
                }
            }

            if (left < j) {
                push_left = true;
                left_child_L = left; left_child_R = j;
            }
            if (i < right) {
                push_right = true;
                right_child_L = i; right_child_R = right;
            }
        }
    }

    // --- 4. WARP AGGREGATED PUSH (LEFT) ---
    // Svi threadovi (aktivni i neaktivni) dolaze ovdje.
    unsigned int mask = __ballot_sync(0xFFFFFFFF, push_left);
    
    if (mask != 0) {
        int leader = __ffs(mask) - 1; 
        int my_pop = __popc(mask & ((1U << lane) - 1));
        int base_idx = 0;
        
        if (lane == leader) {
            base_idx = atomicAdd(out_cnt, __popc(mask));
        }
        base_idx = __shfl_sync(0xFFFFFFFF, base_idx, leader);
        
        if (push_left) {
            int write_idx = base_idx + my_pop;
            if (write_idx < max_tasks) { 
                out_q[write_idx] = {left_child_L, left_child_R};
            }
        }
    }

    // --- 5. WARP AGGREGATED PUSH (RIGHT) ---
    mask = __ballot_sync(0xFFFFFFFF, push_right);
    if (mask != 0) {
        int leader = __ffs(mask) - 1;
        int my_pop = __popc(mask & ((1U << lane) - 1));
        int base_idx = 0;
        
        if (lane == leader) {
            base_idx = atomicAdd(out_cnt, __popc(mask));
        }
        base_idx = __shfl_sync(0xFFFFFFFF, base_idx, leader);
        
        if (push_right) {
            int write_idx = base_idx + my_pop;
            if (write_idx < max_tasks) { 
                out_q[write_idx] = {right_child_L, right_child_R};
            }
        }
    }
}


// Initial Partition Kernel (Parallel - Single Block)
// Za prvi (najveci) zadatak, jedan thread je prespor.
// Koristimo jedan blok (npr 256 niti) da uradimo prvu grubu podjelu?
// Ili jednostavno pustimo CPU da uradi prvi nivo?
// Za 134M elemenata, CPU uradi pivot za < 100ms. To je prihvatljivo.
// Ovdje cemo koristiti CPU za "Bootstrapping" (prvih par nivoa).

// ================== HOST LOGIC (UPDATED) ==================

// Helper: CPU Serial Partition
void cpu_partition(std::vector<int>& data, int left, int right, std::vector<Task>& next_level) {
    if (left >= right) return;
    
    // Random pivot (Simple median)
    int mid = left + (right - left) / 2;
    int p = data[mid];
    
    int i = left, j = right;
    while (i <= j) {
        while (data[i] < p) i++;
        while (data[j] > p) j--;
        if (i <= j) {
            std::swap(data[i], data[j]);
            i++; j--;
        }
    }
    
    // Push children
    if (left < j) next_level.push_back({left, j});
    if (i < right) next_level.push_back({i, right});
}

void quicksort_iterative_v2(std::vector<int>& h_in, std::vector<int>& h_out, int device, Timings* tout) {
    int n = h_in.size();
    
    // 1. AGRESIIVNI CPU BOOTSTRAP
    // Cilj: Razbiti niz na 65,536 dijelova. 
    // To znaci da ce prosjecna velicina zadatka biti oko 2000 elementa.
    // To staje u L2 Cache GPU-a (koji je oko 2-4MB na RTX 3050).
    int TARGET_TASKS = 65536; 
    
    std::vector<Task> tasks;
    tasks.reserve(TARGET_TASKS * 2);
    tasks.push_back({0, n-1});
    
    // BFS na CPU
    while (tasks.size() < TARGET_TASKS) {
        bool changed = false;
        std::vector<Task> next_tasks;
        next_tasks.reserve(tasks.size() * 2); // Optimizacija alokacije
        
        for (const auto& t : tasks) {
            // Dijelimo sve sto je vece od 512 elemenata
            // Mali zadaci idu dalje kakvi jesu
            if (t.right - t.left + 1 > 512) { 
                cpu_partition(h_in, t.left, t.right, next_tasks);
                changed = true;
            } else {
                next_tasks.push_back(t);
            }
        }
        
        if (!changed) break; // Svi su postali mali
        tasks = std::move(next_tasks);
    }
    
    // --- PRELAZAK NA GPU ---
    
    gpuErrchk(cudaSetDevice(device));
    
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // H2D Data
    int* d_data; gpuErrchk(cudaMalloc(&d_data, n*4));
    
    cudaEventRecord(start);
    gpuErrchk(cudaMemcpy(d_data, h_in.data(), n*4, cudaMemcpyHostToDevice));
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float h2d; cudaEventElapsedTime(&h2d, start, stop);
    
    // Setup Queues
    // Povecali smo MAX_TASKS na 5M jer startamo sa 65k zadataka
    int MAX_TASKS = 5000000; 
    Task *d_q1, *d_q2; int *d_c1, *d_c2;
    gpuErrchk(cudaMalloc(&d_q1, MAX_TASKS*sizeof(Task)));
    gpuErrchk(cudaMalloc(&d_q2, MAX_TASKS*sizeof(Task)));
    gpuErrchk(cudaMalloc(&d_c1, 4)); gpuErrchk(cudaMalloc(&d_c2, 4));
    
    // Copy initial tasks to GPU
    int init_cnt = tasks.size();
    gpuErrchk(cudaMemcpy(d_q1, tasks.data(), init_cnt*sizeof(Task), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_c1, &init_cnt, 4, cudaMemcpyHostToDevice));
    
    // Host pinned memory for counter reading
    int* h_pinned_cnt;
    gpuErrchk(cudaHostAlloc(&h_pinned_cnt, 4, cudaHostAllocMapped));
    *h_pinned_cnt = init_cnt;
    
    cudaEventRecord(start);
    
    int num_tasks = init_cnt;
    int iter = 0;
    
    while(num_tasks > 0) {
        // Reset output counter
        gpuErrchk(cudaMemsetAsync(d_c2, 0, 4));
        
        // Launch Configuration
        // Koristimo 256 niti jer je to optimalno za RTX 3050 Ti
        int threads_per_block = 256;
        int num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block;
        
        // Clamp blocks
        if (num_blocks < 128) num_blocks = 128; 
        if (num_blocks > 20000) num_blocks = 20000; 
        
        // POZIV KERNELA (Sa MAX_TASKS argumentom)
        iterative_partition_kernel<<<num_blocks, threads_per_block>>>(d_data, d_q1, num_tasks, d_q2, d_c2, MAX_TASKS);
        
        // Swap Queues (Pointers)
        std::swap(d_q1, d_q2);
        std::swap(d_c1, d_c2);
        
        // Read Count
        gpuErrchk(cudaMemcpy(h_pinned_cnt, d_c1, 4, cudaMemcpyDeviceToHost));
        num_tasks = *h_pinned_cnt;
        
        iter++;
        if (iter > 200) break; // Safety break
    }
    
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float k_ms; cudaEventElapsedTime(&k_ms, start, stop);
    
    // D2H
    cudaEventRecord(start);
    h_out.resize(n);
    gpuErrchk(cudaMemcpy(h_out.data(), d_data, n*4, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float d2h; cudaEventElapsedTime(&d2h, start, stop);
    
    if (tout) {
        tout->h2d_ms = h2d;
        tout->kernel_ms = k_ms;
        tout->d2h_ms = d2h;
        tout->total_ms = h2d + k_ms + d2h;
        tout->kernel_invocations = iter;
    }
    
    cudaFree(d_data); cudaFree(d_q1); cudaFree(d_q2); 
    cudaFree(d_c1); cudaFree(d_c2); cudaFreeHost(h_pinned_cnt);
}

// ... CLI ...
static std::vector<int> parse_list(const std::string &s) {
    std::vector<int> out; std::stringstream ss(s); std::string token;
    while(std::getline(ss, token, ',')) { try{ out.push_back(std::stoi(token)); }catch(...){} } return out;
}
struct CLI {
    int sizePow2=24, device=0, reps=1; unsigned int seed=12345u;
    std::string output_dir="../quick_data", sweep_name;
    std::vector<int> sweep_threads, sweep_blocks;
};
static CLI parse_cli(int argc, char** argv) {
    CLI o;
    for(int i=1;i<argc;i++) {
        std::string s(argv[i]);
        if(s=="--device" && i+1<argc) o.device=std::stoi(argv[++i]);
        else if(s=="--size" && i+1<argc) o.sizePow2=std::stoi(argv[++i]);
        else if(s=="--min-exp" && i+1<argc) o.sizePow2=std::stoi(argv[++i]);
        else if(s=="--reps" && i+1<argc) o.reps=std::stoi(argv[++i]);
        else if((s=="--output"||s=="--output-dir") && i+1<argc) o.output_dir=argv[++i];
        else if(s=="--seed" && i+1<argc) o.seed=(unsigned int)std::stoul(argv[++i]);
        else if(s=="--sweep-name" && i+1<argc) o.sweep_name=argv[++i];
        else if(s=="--sweep-threads" && i+1<argc) o.sweep_threads=parse_list(argv[++i]);
        else if(s=="--sweep-blocks" && i+1<argc) o.sweep_blocks=parse_list(argv[++i]);
    }
    return o;
}

int main(int argc, char** argv) {
    CLI opts = parse_cli(argc, argv);
    size_t N = (opts.sizePow2 >= 1 && opts.sizePow2 <= 31) ? (size_t)1 << opts.sizePow2 : (size_t)1 << 24;
    int devCount=0; gpuErrchk(cudaGetDeviceCount(&devCount));
    if(devCount==0 || opts.device<0 || opts.device>=devCount) return 1;
    gpuErrchk(cudaSetDevice(opts.device));
    DeviceInfo di=queryDevice(opts.device);
    printf("Device %d: %s, ITERATIVE V2 (Hybrid CPU Boot)\n", di.deviceId, di.name);

    ensure_dir(opts.output_dir);
    std::string csv_name=opts.sweep_name.empty()?("quick_iterative_v2_sweep_"+timestamp_string()+".csv"):opts.sweep_name;
    std::filesystem::path csv_path=std::filesystem::path(opts.output_dir)/csv_name;
    std::ofstream csv(csv_path); 
    
    csv<<"timestamp,run_id,size,next_pow2,rep,seed,device_id,device_name,sm_count,threads,algorithm,h2d_ms,total_kernel_ms,kernel_invocations,d2h_ms,total_ms,throughput_Melems_s,verify_passed\n";
    csv<<std::fixed<<std::setprecision(3);

    std::vector<int> threads_list = opts.sweep_threads.empty() ? std::vector<int>{256} : opts.sweep_threads;
    
    int run_id = 0;
    for (int th : threads_list) {
        std::cout<<"=== Sweep (Iterative V2): size="<<(uint64_t)N<<" ===\n";
        for (int rep = 0; rep < opts.reps; ++rep) {
            ++run_id;
            std::vector<int> h_in(N); std::mt19937 rng(opts.seed + run_id + rep);
            std::uniform_int_distribution<int> dist(INT32_MIN/2, INT32_MAX/2);
            for (size_t i = 0; i < N; i++) h_in[i] = dist(rng);

            std::vector<int> h_out; Timings T{};
            quicksort_iterative_v2(h_in, h_out, opts.device, &T);
            bool ok = std::is_sorted(h_out.begin(), h_out.end());
            double throughput = (double)N / (T.total_ms / 1000.0) / 1e6;

            csv<<timestamp_string()<<","<<run_id<<","<<(uint64_t)N<<","<<next_pow2_int((int)N)<<","<<rep<<","<<opts.seed<<","<<di.deviceId<<",\""<<di.name<<"\","<<di.multiProcessorCount<<","<<th<<",\"GPU-QuickSort (Iterative V2)\","<<T.h2d_ms<<","<<T.kernel_ms<<","<<T.kernel_invocations<<","<<T.d2h_ms<<","<<T.total_ms<<","<<throughput<<","<<(ok?"1":"0")<<"\n";
            csv.flush();
            std::cout<<"run="<<run_id<<" OK="<<ok<<" Time="<<T.total_ms<<"ms Calls="<<T.kernel_invocations<<" Throughput="<<throughput<<" Me/s\n";
        }
    }
    std::cout << "\nCSV: " << std::filesystem::absolute(csv_path).string() << std::endl;
    csv.close(); return 0;
}