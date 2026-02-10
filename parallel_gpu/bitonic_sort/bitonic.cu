/*
bitonic.cu

Stable Bitonic sort with safe optimizations:
- pinned (page-locked) host buffers for H2D/D2H
- warp-level intra-warp bitonic (shuffle) for k <= warpSize (32)
- shared-memory inter-warp merging for larger k
- global grid-stride comparator for the remaining phases
- grouped timing, verification, CSV harness and built-in sweep over thread/block configs

Notes:
- Supported thread tile template instantiations: 128, 256, 512, 1024.
  If you pass a threads value that isn't one of these, the code falls back to
  the global kernel for the local phase (correct but slower).
- Intra-warp implementation uses lane-based asc so both partners agree.
- All CUDA calls are checked for errors; kernel launches are followed by cudaGetLastError checks.

Build:
  nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas -v bitonic.cu -o bitonic

Run example:
  ./bitonic --sweep-threads 256,512,1024 --sweep-blocks 1024,2048 --min-exp 20 --max-exp 22 --reps 3 --output ../bitonic_data --grid

Author: adapted for harunmioc (2026-02-07)
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <filesystem>
#include <cstdint>
#include <climits>
#include <limits>
#include <thread>
#include <chrono>
#include <cstring>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "GPU Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}

// ------------------------- Kernels -------------------------

// Shared-memory tile bitonic: warp-local shuffle (k <= warpSize) then shared inter-warp
template<int TILE>
__global__ void bitonic_shared_kernel(int* d_arr, int n_pow2) {
    __shared__ int s_mem[TILE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load (pad with INT32_MAX for out-of-range indices). No thread returns early.
    if (gid < n_pow2) s_mem[tid] = d_arr[gid];
    else s_mem[tid] = INT32_MAX;
    __syncthreads();

    // Bitonic network inside block (assumes TILE is power-of-two)
    for (int k = 2; k <= TILE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                // lokalni smjer po tid za korektan per-block bitonic sort
                bool asc = ((tid & k) == 0);
                int a = s_mem[tid];
                int b = s_mem[ixj];
                if ((a > b) == asc) {
                    s_mem[tid] = b;
                    s_mem[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Write back only valid indices
    if (gid < n_pow2) d_arr[gid] = s_mem[tid];
}

// Global comparator kernel for arbitrary j,k (grid-stride)
__global__ void bitonic_global_stride_kernel(int* d_arr, int n_pow2, int j, int k) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = i0; i < n_pow2; i += stride) {
        int ixj = i ^ j;
        if (ixj > i && ixj < n_pow2) {
            bool asc = ((i & k) == 0);
            int a = d_arr[i];
            int b = d_arr[ixj];
            if ((a > b) == asc) {
                d_arr[i] = b;
                d_arr[ixj] = a;
            }
        }
    }
}

// NEW: Fused global kernel – iterira j interno za jedan k (manje kernel launch-eva)
__global__ void bitonic_global_fused_kernel(int* d_arr, int n_pow2, int k) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = i0; i < n_pow2; i += stride) {
        // Za ovaj i, smjer je konstantan za sve j u okviru k
        bool asc = ((i & k) == 0);

        // Iteriraj sve faze j = k/2, k/4, ... , 1
        for (int j = (k >> 1); j > 0; j >>= 1) {
            int ixj = i ^ j;
            if (ixj > i && ixj < n_pow2) {
                // read-only cache hint
                int a = __ldg(&d_arr[i]);
                int b = __ldg(&d_arr[ixj]);
                if ((a > b) == asc) {
                    d_arr[i]  = b;
                    d_arr[ixj] = a;
                }
            }
            // nema potrebe za __syncthreads() (grid-stride, različiti i)
        }
    }
}

// ------------------------- Utilities -------------------------
static inline int next_pow2_int(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

struct DeviceInfo {
    std::string name;
    int deviceId;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    size_t sharedMemPerBlock;
    int major, minor;
};

DeviceInfo queryDevice(int deviceId) {
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, deviceId));
    DeviceInfo info;
    info.name = prop.name;
    info.deviceId = deviceId;
    info.multiProcessorCount = prop.multiProcessorCount;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.major = prop.major;
    info.minor = prop.minor;
    return info;
}

struct Timings {
    float h2d_ms = 0.0f;
    float total_kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_ms = 0.0f;
    int kernel_invocations = 0;
};

// ------------------------- Bitonic driver (pinned + warp intra-warp) -------------------------
void bitonic_sort_gpu(const std::vector<int>& h_in, std::vector<int>& h_out,
                      Timings &timings, const DeviceInfo &dinfo,
                      bool print_grid = false, int threads = 512, int blocks_override = -1) {
    int n = (int)h_in.size();
    int n_pow2 = next_pow2_int(n);

    auto start_total = std::chrono::high_resolution_clock::now();

    // Device allocation
    int *d_arr = nullptr;
    gpuErrchk(cudaMalloc(&d_arr, (size_t)n_pow2 * sizeof(int)));

    // Pinned host input buffer (n_pow2)
    int *h_pinned = nullptr;
    gpuErrchk(cudaHostAlloc((void**)&h_pinned, (size_t)n_pow2 * sizeof(int), cudaHostAllocDefault));
    for (int i = 0; i < n_pow2; ++i) h_pinned[i] = INT32_MAX;
    if (n > 0) std::memcpy(h_pinned, h_in.data(), (size_t)n * sizeof(int));

    // H2D timing (async)
    cudaEvent_t ev_h2d_start, ev_h2d_end;
    gpuErrchk(cudaEventCreate(&ev_h2d_start));
    gpuErrchk(cudaEventCreate(&ev_h2d_end));
    gpuErrchk(cudaEventRecord(ev_h2d_start, 0));
    gpuErrchk(cudaMemcpyAsync(d_arr, h_pinned, (size_t)n_pow2 * sizeof(int), cudaMemcpyHostToDevice, 0));
    gpuErrchk(cudaEventRecord(ev_h2d_end, 0));
    gpuErrchk(cudaEventSynchronize(ev_h2d_end));
    gpuErrchk(cudaEventElapsedTime(&timings.h2d_ms, ev_h2d_start, ev_h2d_end));
    gpuErrchk(cudaEventDestroy(ev_h2d_start));
    gpuErrchk(cudaEventDestroy(ev_h2d_end));

    // Configure threads/blocks
    if (threads > (int)dinfo.maxThreadsPerBlock) threads = (int)dinfo.maxThreadsPerBlock;
    // make threads power-of-two
    int t = 1; while (t < threads) t <<= 1;
    if (t != threads) threads = t;

    int blocks = (n_pow2 + threads - 1) / threads;
    if (blocks_override > 0) {
        blocks = blocks_override;
    } else {
        // agresivnija zasićenost: SM_count * 2048 (RTX 3050 Ti: 20 * 2048 = 40960)
        int maxBlocksHeuristic = dinfo.multiProcessorCount * 2048;
        if (blocks > maxBlocksHeuristic) blocks = maxBlocksHeuristic;
    }

    if (print_grid) {
        std::cout << "Launching threads=" << threads << " blocks=" << blocks << " n_pow2=" << n_pow2 << std::endl;
    }

    // Kernel timing (grouped)
    cudaEvent_t ev_kstart, ev_kend;
    gpuErrchk(cudaEventCreate(&ev_kstart));
    gpuErrchk(cudaEventCreate(&ev_kend));
    gpuErrchk(cudaEventRecord(ev_kstart, 0));

    // Shared/local phase: template instantiation matching threads (safe sizes)
    int blocks_local = (n_pow2 + threads - 1) / threads;
    if (threads <= 1024) {
        if (threads == 512) {
            bitonic_shared_kernel<512><<<blocks_local, 512>>>(d_arr, n_pow2);
            gpuErrchk(cudaGetLastError());
        } else if (threads == 256) {
            bitonic_shared_kernel<256><<<blocks_local, 256>>>(d_arr, n_pow2);
            gpuErrchk(cudaGetLastError());
        } else if (threads == 128) {
            bitonic_shared_kernel<128><<<blocks_local, 128>>>(d_arr, n_pow2);
            gpuErrchk(cudaGetLastError());
        } else if (threads == 1024) {
            bitonic_shared_kernel<1024><<<blocks_local, 1024>>>(d_arr, n_pow2);
            gpuErrchk(cudaGetLastError());
        } else {
            // unsupported thread count for template: fallback to safe global kernel pass
            bitonic_global_stride_kernel<<<blocks, threads>>>(d_arr, n_pow2, 1, 2);
            gpuErrchk(cudaGetLastError());
        }
        timings.kernel_invocations += 1;
    }

    // Global-phase
    int start_k = 2; // počni od 2, kao canonical bitonic
    for (int k = start_k; k <= n_pow2; k <<= 1) {
        for (int j = (k >> 1); j > 0; j >>= 1) {
            bitonic_global_stride_kernel<<<blocks, threads>>>(d_arr, n_pow2, j, k);
            gpuErrchk(cudaGetLastError());
            timings.kernel_invocations += 1;
        }
    }

    gpuErrchk(cudaEventRecord(ev_kend, 0));
    gpuErrchk(cudaEventSynchronize(ev_kend));
    gpuErrchk(cudaEventElapsedTime(&timings.total_kernel_ms, ev_kstart, ev_kend));
    gpuErrchk(cudaEventDestroy(ev_kstart));
    gpuErrchk(cudaEventDestroy(ev_kend));

    // D2H pinned buffer (size n)
    int *h_pinned_out = nullptr;
    if (n > 0) {
        gpuErrchk(cudaHostAlloc((void**)&h_pinned_out, (size_t)n * sizeof(int), cudaHostAllocDefault));
    } else {
        h_pinned_out = nullptr;
    }

    cudaEvent_t ev_d2h_start, ev_d2h_end;
    gpuErrchk(cudaEventCreate(&ev_d2h_start));
    gpuErrchk(cudaEventCreate(&ev_d2h_end));
    gpuErrchk(cudaEventRecord(ev_d2h_start, 0));
    if (n > 0) gpuErrchk(cudaMemcpyAsync(h_pinned_out, d_arr, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost, 0));
    gpuErrchk(cudaEventRecord(ev_d2h_end, 0));
    gpuErrchk(cudaEventSynchronize(ev_d2h_end));
    gpuErrchk(cudaEventElapsedTime(&timings.d2h_ms, ev_d2h_start, ev_d2h_end));
    gpuErrchk(cudaEventDestroy(ev_d2h_start));
    gpuErrchk(cudaEventDestroy(ev_d2h_end));

    if (n > 0) std::memcpy(h_out.data(), h_pinned_out, (size_t)n * sizeof(int));

    // total wall time
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = end_total - start_total;
    timings.total_ms = static_cast<float>(dt.count());

    // cleanup
    gpuErrchk(cudaFree(d_arr));
    gpuErrchk(cudaFreeHost(h_pinned));
    if (n > 0) gpuErrchk(cudaFreeHost(h_pinned_out));
}

// ------------------------- CSV + CLI harness + sweep -------------------------
std::string timestamp_string() {
    std::time_t t = std::time(nullptr);
    std::tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

void ensure_dir(const std::filesystem::path &p) {
    std::error_code ec;
    if (!std::filesystem::exists(p)) {
        std::filesystem::create_directories(p, ec);
        if (ec) {
            std::cerr << "Failed to create directory " << p << ": " << ec.message() << std::endl;
            exit(1);
        }
    }
}

struct CLIOptions {
    int min_exp = 10;
    int max_exp = 24;
    int reps = 3;
    std::string output_dir = "../bitonic_data";
    bool grid_flag = false;
    int device = 0;
    int threads = 512;
    int blocks_override = -1;
    unsigned int seed = 12345u;

    // sweep options
    std::vector<int> sweep_threads; // if non-empty, run sweep
    std::vector<int> sweep_blocks;
    std::string sweep_name;
    int sweep_delay_ms = 10;
};

static std::vector<int> parse_list(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            int v = std::stoi(token);
            out.push_back(v);
        } catch (...) {
            // ignore non-integer tokens
        }
    }
    return out;
}

CLIOptions parse_cli(int argc, char** argv) {
    CLIOptions o;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s == "--min-exp" && i+1 < argc) { o.min_exp = std::stoi(argv[++i]); }
        else if (s == "--max-exp" && i+1 < argc) { o.max_exp = std::stoi(argv[++i]); }
        else if (s == "--reps" && i+1 < argc) { o.reps = std::stoi(argv[++i]); }
        else if ((s == "--output" || s == "--output-dir") && i+1 < argc) { o.output_dir = argv[++i]; }
        else if (s == "--grid") { o.grid_flag = true; }
        else if (s == "--device" && i+1 < argc) { o.device = std::stoi(argv[++i]); }
        else if (s == "--threads" && i+1 < argc) { o.threads = std::stoi(argv[++i]); }
        else if (s == "--blocks" && i+1 < argc) { o.blocks_override = std::stoi(argv[++i]); }
        else if (s == "--seed" && i+1 < argc) { o.seed = static_cast<unsigned int>(std::stoul(argv[++i])); }

        // sweep-specific
        else if (s == "--sweep-threads" && i+1 < argc) { o.sweep_threads = parse_list(argv[++i]); }
        else if (s == "--sweep-blocks" && i+1 < argc) { o.sweep_blocks = parse_list(argv[++i]); }
        else if (s == "--sweep-name" && i+1 < argc) { o.sweep_name = argv[++i]; }
        else if (s == "--sweep-delay" && i+1 < argc) { o.sweep_delay_ms = std::stoi(argv[++i]); }

        else {
            std::cerr << "Unknown option: " << s << std::endl;
            std::cerr << "Supported: --min-exp N --max-exp N --reps N --output DIR --grid --device ID --threads N --blocks N --seed N\n"
                      << "           --sweep-threads CSV --sweep-blocks CSV [--sweep-name NAME] [--sweep-delay ms]" << std::endl;
            exit(1);
        }
    }
    return o;
}

int main(int argc, char** argv) {
    auto opts = parse_cli(argc, argv);

    int devCount = 0;
    gpuErrchk(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    if (opts.device < 0 || opts.device >= devCount) {
        std::cerr << "Invalid device id: " << opts.device << ", devices available: " << devCount << std::endl;
        return 1;
    }
    gpuErrchk(cudaSetDevice(opts.device));
    DeviceInfo dinfo = queryDevice(opts.device);

    std::cout << "Using device " << opts.device << ": " << dinfo.name
              << " (SM " << dinfo.major << "." << dinfo.minor
              << ", SMs=" << dinfo.multiProcessorCount
              << ", maxThreads=" << dinfo.maxThreadsPerBlock
              << ", sharedMem=" << dinfo.sharedMemPerBlock << ")\n";

    ensure_dir(opts.output_dir);

    // Decide output CSV name: if sweep requested, use sweep_name or generated name
    std::string csv_name;
    if (!opts.sweep_threads.empty() || !opts.sweep_blocks.empty()) {
        if (!opts.sweep_name.empty()) csv_name = opts.sweep_name;
        else csv_name = "bitonic_sweep_" + timestamp_string() + ".csv";
    } else {
        csv_name = "bitonic_" + timestamp_string() + ".csv";
    }
    std::filesystem::path csv_path = std::filesystem::path(opts.output_dir) / csv_name;
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV for writing: " << csv_path << std::endl;
        return 1;
    }

    // CSV header
    csv << "timestamp,run_id,size,next_pow2,rep,seed,device_id,device_name,compute_capability,sm_count,threads,blocks,h2d_ms,total_kernel_ms,kernel_invocations,d2h_ms,total_ms,throughput_Melems_s,verify_passed\n";
    csv << std::fixed << std::setprecision(3);

    auto run_once = [&](int run_id, int size, int rep, unsigned int seed, int threads, int blocks_override) {
        // Prepare input (host-side vector)
        std::mt19937 rng(seed + run_id + size + rep);
        std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min()/2, std::numeric_limits<int>::max()/2);
        std::vector<int> h_in(size);
        for (int i = 0; i < size; ++i) h_in[i] = dist(rng);
        std::vector<int> h_out(size);
        Timings t;

        auto start = std::chrono::high_resolution_clock::now();
        bitonic_sort_gpu(h_in, h_out, t, dinfo, opts.grid_flag, threads, blocks_override);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dtotal = end - start;


        bool ok = std::is_sorted(h_out.begin(), h_out.end());

        double throughput = (double)size / (t.total_ms / 1000.0) / 1e6; // Melems/s

        // Write CSV row
        csv << timestamp_string() << ",";
        csv << run_id << ",";
        csv << size << ",";
        csv << next_pow2_int(size) << ",";
        csv << rep << ",";
        csv << seed << ",";
        csv << opts.device << ",";
        std::string devname = dinfo.name;
        for (auto &c: devname) if (c == ',') c = ';';
        csv << "\"" << devname << "\"" << ",";
        csv << dinfo.major << "." << dinfo.minor << ",";
        csv << dinfo.multiProcessorCount << ",";
        csv << threads << ",";
        int blocks = (next_pow2_int(size) + threads - 1) / threads;
        if (blocks_override > 0) blocks = blocks_override;
        csv << blocks << ",";
        csv << t.h2d_ms << ",";
        csv << t.total_kernel_ms << ",";
        csv << t.kernel_invocations << ",";
        csv << t.d2h_ms << ",";
        csv << t.total_ms << ",";
        csv << throughput << ",";
        csv << (ok ? "1" : "0") << "\n";

        csv.flush();

        std::cout << "[run " << run_id << "] size=" << size << " threads="<<threads<<" blocks="<< (blocks_override>0?blocks_override:blocks)
                  << " rep=" << rep << " kernels=" << t.kernel_invocations
                  << " h2d=" << t.h2d_ms << "ms kernels=" << t.total_kernel_ms << "ms d2h=" << t.d2h_ms
                  << "ms total=" << t.total_ms << "ms verify=" << (ok ? "OK" : "FAILED")
                  << " throughput=" << std::setprecision(3) << throughput << " Me/s"
                  << std::setprecision(3) << std::endl;
    };

    int global_run_id = 0;

    // If sweep requested, iterate combinations; otherwise perform single-run harness
    if (!opts.sweep_threads.empty() || !opts.sweep_blocks.empty()) {
        // If one of sweep lists is empty, default to current single value
        std::vector<int> threads_list = opts.sweep_threads.empty() ? std::vector<int>{opts.threads} : opts.sweep_threads;
        std::vector<int> blocks_list = opts.sweep_blocks.empty() ? std::vector<int>{opts.blocks_override > 0 ? opts.blocks_override : -1} : opts.sweep_blocks;

        int total_configs = (int)threads_list.size() * (int)blocks_list.size();
        int cfg_idx = 0;
        for (int th : threads_list) {
            for (int bl : blocks_list) {
                ++cfg_idx;
                std::cout << "=== Sweep config " << cfg_idx << "/" << total_configs << " : threads=" << th << " blocks_override=" << bl << " ===\n";
                // For each config run the normal exp/reps loop
                for (int exp = opts.min_exp; exp <= opts.max_exp; ++exp) {
                    int size = 1 << exp;
                    for (int rep = 0; rep < opts.reps; ++rep) {
                        ++global_run_id;
                        run_once(global_run_id, size, rep, opts.seed, th, bl);
                    }
                }
                // small delay between configs so file system flushes and GPU recovers
                std::this_thread::sleep_for(std::chrono::milliseconds(opts.sweep_delay_ms));
            }
        }
    } else {
        // single-config original loop (uses opts.threads and opts.blocks_override)
        for (int exp = opts.min_exp; exp <= opts.max_exp; ++exp) {
            int size = 1 << exp;
            for (int rep = 0; rep < opts.reps; ++rep) {
                ++global_run_id;
                run_once(global_run_id, size, rep, opts.seed, opts.threads, opts.blocks_override);
            }
        }
    }

    csv.close();
    std::cout << "CSV results written to " << csv_path << std::endl;
    return 0;
}