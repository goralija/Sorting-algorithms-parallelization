#pragma once
#include <string>
#include <vector>

struct DeviceInfo {
    std::string name;
    int deviceId;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    size_t sharedMemPerBlock;
    int major, minor;
};

struct Timings {
    float h2d_ms = 0.0f;
    float total_kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_ms = 0.0f;
    int kernel_invocations = 0;
};

// Radix API (potpis koji si već koristio u radix.cu)
void radix_sort_gpu(const std::vector<int>& h_in_signed,
                    std::vector<int>& h_out_signed,
                    Timings &timings,
                    const DeviceInfo &dinfo,
                    int threads,
                    int blocks_override);

// Merge API (adapter koji iz merge.cu treba da izveze stabilan potpis)
void merge_sort_gpu(const std::vector<int>& h_in,
                    std::vector<int>& h_out,
                    Timings &timings,
                    const DeviceInfo &dinfo,
                    int threads,
                    int blocks_override);