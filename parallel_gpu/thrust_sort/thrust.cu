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
#include <thread>

// Zajednički header: tipovi DeviceInfo/Timings + prototipi merge_sort_gpu/radix_sort_gpu (prošireni potpis)
#include "../include/merge_radix_api.hpp"

// Thrust baseline (samo za zvanični poređeni sort; za poređenje)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "GPU Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}

// ------------------------- CLI + CSV helpers -------------------------
static inline int next_pow2_int(int v) { int p=1; while (p<v) p<<=1; return p; }

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

static std::vector<int> parse_list(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { out.push_back(std::stoi(token)); } catch (...) {}
    }
    return out;
}

struct CLIOptions {
    int min_exp = 27;  // default: 2^27
    int max_exp = 27;
    int reps = 3;
    std::string output_dir = "../thrust_data";
    int device = 0;
    int threads = 512;
    int blocks_override = -1;
    unsigned int seed = 12345u;

    std::vector<int> sweep_threads;
    std::vector<int> sweep_blocks;
    std::string sweep_name;
    int sweep_delay_ms = 200;
};

CLIOptions parse_cli(int argc, char** argv) {
    CLIOptions o;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        if (s == "--min-exp" && i+1 < argc) o.min_exp = std::stoi(argv[++i]);
        else if (s == "--max-exp" && i+1 < argc) o.max_exp = std::stoi(argv[++i]);
        else if (s == "--reps" && i+1 < argc) o.reps = std::stoi(argv[++i]);
        else if ((s == "--output" || s == "--output-dir") && i+1 < argc) o.output_dir = argv[++i];
        else if (s == "--device" && i+1 < argc) o.device = std::stoi(argv[++i]);
        else if (s == "--threads" && i+1 < argc) o.threads = std::stoi(argv[++i]);
        else if (s == "--blocks" && i+1 < argc) o.blocks_override = std::stoi(argv[++i]);
        else if (s == "--seed" && i+1 < argc) o.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        else if (s == "--sweep-threads" && i+1 < argc) o.sweep_threads = parse_list(argv[++i]);
        else if (s == "--sweep-blocks" && i+1 < argc) o.sweep_blocks = parse_list(argv[++i]);
        else if (s == "--sweep-name" && i+1 < argc) o.sweep_name = argv[++i];
        else if (s == "--sweep-delay" && i+1 < argc) o.sweep_delay_ms = std::stoi(argv[++i]);
        else {
            std::cerr << "Unknown option: " << s << std::endl;
            exit(1);
        }
    }
    return o;
}

// ------------------------- Device info -------------------------
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

// ------------------------- Hybrid sort + Thrust baseline -------------------------
// prag pragmatičan prag za CPU/merge/radix
static constexpr std::size_t kCpuThreshold = 2048;
static constexpr std::size_t kMergeMaxSize = 1'000'000;

// Koristi tvoje GPU funkcije (merge_sort_gpu / radix_sort_gpu) sa proširenim potpisom
// DODANO: out.resize(...) da verifikacija bude ispravna
void custom_hybrid_sort(std::vector<int>& h_arr, int threads, int blocks_override, int deviceId, std::string &which_alg) {
    const std::size_t n = h_arr.size();
    if (n == 0) { which_alg = "CustomHybrid(Empty)"; return; }

    if (n <= kCpuThreshold) {
        std::sort(h_arr.begin(), h_arr.end());
        which_alg = "CustomHybrid(CPU)";
        return;
    }

    gpuErrchk(cudaSetDevice(deviceId));
    DeviceInfo dinfo = queryDevice(deviceId);

    Timings t{};
    std::vector<int> out;
    out.resize(h_arr.size()); // critical fix: we must size the output vector

    if (n <= kMergeMaxSize) {
        which_alg = "CustomHybrid(Merge)";
        merge_sort_gpu(h_arr, out, t, dinfo, threads, blocks_override);
    } else {
        which_alg = "CustomHybrid(Radix)";
        radix_sort_gpu(h_arr, out, t, dinfo, threads, blocks_override);
    }
    h_arr.swap(out);
}

// Official Thrust baseline s GPU event timing (H2D/sort/D2H)
void official_thrust_sort_timed(const std::vector<int>& h_in, std::vector<int>& h_out, Timings &timings) {
    const int n = static_cast<int>(h_in.size());
    h_out.resize(n);

    int* d_data = nullptr;
    gpuErrchk(cudaMalloc(&d_data, (size_t)n * sizeof(int)));

    cudaEvent_t ev_h2d_start, ev_h2d_end, ev_kstart, ev_kend, ev_d2h_start, ev_d2h_end;
    gpuErrchk(cudaEventCreate(&ev_h2d_start));
    gpuErrchk(cudaEventCreate(&ev_h2d_end));
    gpuErrchk(cudaEventCreate(&ev_kstart));
    gpuErrchk(cudaEventCreate(&ev_kend));
    gpuErrchk(cudaEventCreate(&ev_d2h_start));
    gpuErrchk(cudaEventCreate(&ev_d2h_end));

    gpuErrchk(cudaEventRecord(ev_h2d_start, 0));
    gpuErrchk(cudaMemcpyAsync(d_data, h_in.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice, 0));
    gpuErrchk(cudaEventRecord(ev_h2d_end, 0));
    gpuErrchk(cudaEventSynchronize(ev_h2d_end));
    gpuErrchk(cudaEventElapsedTime(&timings.h2d_ms, ev_h2d_start, ev_h2d_end));

    gpuErrchk(cudaEventRecord(ev_kstart, 0));
    {
        thrust::device_ptr<int> dev_begin(d_data);
        thrust::device_ptr<int> dev_end(d_data + n);
        thrust::sort(dev_begin, dev_end);
        timings.kernel_invocations += 1;
    }
    gpuErrchk(cudaEventRecord(ev_kend, 0));
    gpuErrchk(cudaEventSynchronize(ev_kend));
    gpuErrchk(cudaEventElapsedTime(&timings.total_kernel_ms, ev_kstart, ev_kend));

    gpuErrchk(cudaEventRecord(ev_d2h_start, 0));
    gpuErrchk(cudaMemcpyAsync(h_out.data(), d_data, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost, 0));
    gpuErrchk(cudaEventRecord(ev_d2h_end, 0));
    gpuErrchk(cudaEventSynchronize(ev_d2h_end));
    gpuErrchk(cudaEventElapsedTime(&timings.d2h_ms, ev_d2h_start, ev_d2h_end));

    timings.total_ms = timings.h2d_ms + timings.total_kernel_ms + timings.d2h_ms;

    gpuErrchk(cudaEventDestroy(ev_h2d_start));
    gpuErrchk(cudaEventDestroy(ev_h2d_end));
    gpuErrchk(cudaEventDestroy(ev_kstart));
    gpuErrchk(cudaEventDestroy(ev_kend));
    gpuErrchk(cudaEventDestroy(ev_d2h_start));
    gpuErrchk(cudaEventDestroy(ev_d2h_end));
    gpuErrchk(cudaFree(d_data));
}

// ------------------------- CSV + sweep harness -------------------------
int main(int argc, char** argv) {
    // Warm up CUDA context
    cudaFree(nullptr);

    CLIOptions opts = parse_cli(argc, argv);

    int devCount = 0;
    gpuErrchk(cudaGetDeviceCount(&devCount));
    if (devCount == 0) { std::cerr << "No CUDA devices found.\n"; return 1; }
    if (opts.device < 0 || opts.device >= devCount) {
        std::cerr << "Invalid device id: " << opts.device << ", devices available: " << devCount << std::endl;
        return 1;
    }
    gpuErrchk(cudaSetDevice(opts.device));

    // CSV file
    ensure_dir(opts.output_dir);
    std::string csv_name = opts.sweep_name.empty()
        ? (opts.sweep_threads.empty() && opts.sweep_blocks.empty()
            ? "thrust_" + timestamp_string() + ".csv"
            : "thrust_sweep_" + timestamp_string() + ".csv")
        : opts.sweep_name;

    std::filesystem::path csv_path = std::filesystem::path(opts.output_dir) / csv_name;
    std::ofstream csv(csv_path);
    if (!csv.is_open()) { std::cerr << "Failed to open CSV: " << csv_path << std::endl; return 1; }

    // DODANO: 'algorithm' kolona u CSV-u
    csv << "timestamp,run_id,size,next_pow2,rep,seed,device_id,device_name,compute_capability,sm_count,threads,blocks,algorithm,"
           "h2d_ms,total_kernel_ms,kernel_invocations,d2h_ms,total_ms,throughput_Melems_s,verify_passed\n";
    csv << std::fixed << std::setprecision(3);

    auto write_row = [&](int run_id, int size, int rep, int threads, int blocks, const std::string& algorithm, const Timings &t, bool ok) {
        double throughput = (double)size / (t.total_ms / 1000.0) / 1e6;
        // Device name
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, opts.device);
        std::string devname = prop.name; for (auto &c: devname) if (c == ',') c = ';';

        csv << timestamp_string() << "," << run_id << "," << size << "," << next_pow2_int(size) << ","
            << rep << "," << opts.seed << "," << opts.device << ","
            << "\"" << devname << "\"" << "," << prop.major << "." << prop.minor << ","
            << prop.multiProcessorCount << "," << threads << "," << blocks << ","
            << "\"" << algorithm << "\"" << ","
            << t.h2d_ms << "," << t.total_kernel_ms << "," << t.kernel_invocations << ","
            << t.d2h_ms << "," << t.total_ms << "," << throughput << "," << (ok ? "1" : "0") << "\n";
        csv.flush();
    };

    auto run_once_both = [&](int run_id, int size, int rep, unsigned int seed, int threads, int blocks_override) {
        std::cout << "[start] run_id=" << run_id
                  << " size=" << size
                  << " rep=" << rep
                  << " threads=" << threads
                  << " blocks_override=" << blocks_override
                  << std::endl;

        std::mt19937 rng(seed + run_id + size + rep);
        std::uniform_int_distribution<int> dist(std::numeric_limits<int>::min()/2, std::numeric_limits<int>::max()/2);
        std::vector<int> h_in(size);
        for (int i = 0; i < size; ++i) h_in[i] = dist(rng);

        int natural_blocks = (size + threads - 1) / threads;
        int blocks = (blocks_override > 0 && blocks_override >= natural_blocks) ? blocks_override : natural_blocks;

        // Custom hybrid (wall-clock)
        {
            std::vector<int> h_custom = h_in;
            Timings t{};
            std::string which_alg;
            auto host_start = std::chrono::high_resolution_clock::now();
            custom_hybrid_sort(h_custom, threads, blocks_override, opts.device, which_alg);
            auto host_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dt = host_end - host_start;
            t.total_ms = (float)dt.count();
            t.total_kernel_ms = t.total_ms; // treat as total for throughput comparison
            t.h2d_ms = 0.0f;
            t.d2h_ms = 0.0f;
            t.kernel_invocations = -1;

            std::vector<int> h_ref = h_in;
            std::sort(h_ref.begin(), h_ref.end());
            bool ok = (h_ref == h_custom);

            std::cout << "[custom] run_id=" << run_id
                      << " alg=" << which_alg
                      << " total_ms=" << t.total_ms
                      << " verify=" << (ok ? "OK" : "FAILED")
                      << std::endl;

            write_row(run_id, size, rep, threads, blocks, which_alg, t, ok);
        }

        // Official Thrust (device events)
        {
            std::vector<int> h_thrust_out;
            Timings t{};
            official_thrust_sort_timed(h_in, h_thrust_out, t);

            std::vector<int> h_ref = h_in;
            std::sort(h_ref.begin(), h_ref.end());
            bool ok = (h_ref == h_thrust_out);

            std::string which_alg = "ThrustBaseline";
            std::cout << "[thrust] run_id=" << run_id
                      << " total_ms=" << t.total_ms
                      << " verify=" << (ok ? "OK" : "FAILED")
                      << std::endl;

            write_row(run_id, size, rep, threads, blocks, which_alg, t, ok);
        }
    };

    int global_run_id = 0;
    std::vector<int> threads_list = opts.sweep_threads.empty() ? std::vector<int>{opts.threads} : opts.sweep_threads;
    std::vector<int> blocks_list  = opts.sweep_blocks.empty()  ? std::vector<int>{opts.blocks_override > 0 ? opts.blocks_override : -1}
                                                                : opts.sweep_blocks;
    int total_configs = (int)threads_list.size() * (int)blocks_list.size();
    int cfg_idx = 0;

    for (int th : threads_list) {
        for (int bl : blocks_list) {
            ++cfg_idx;
            std::cout << "=== Sweep config " << cfg_idx << "/" << total_configs
                      << " : threads=" << th << " blocks_override=" << bl << " ===\n";
            for (int exp = opts.min_exp; exp <= opts.max_exp; ++exp) {
                int size = 1 << exp;
                for (int rep = 0; rep < opts.reps; ++rep) {
                    ++global_run_id;
                    run_once_both(global_run_id, size, rep, opts.seed, th, bl);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(opts.sweep_delay_ms));
        }
    }

    csv.close();
    std::cout << "CSV results written to " << (std::filesystem::absolute(csv_path)).string() << std::endl;
    return 0;
}