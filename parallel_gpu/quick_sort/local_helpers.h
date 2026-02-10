#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef GIGA
#define GIGA (1024.0*1024.0*1024.0)
#endif

// Minimal check macro replacement
#ifndef checkCudaErrors
#define checkCudaErrors(val) do { \
    cudaError_t err__ = (val); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #val, __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit((int)err__); \
    } \
} while(0)
#endif

// Minimal getLastCudaError replacement (to check after kernel launches)
inline void getLastCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit((int)err);
    }
}

// SDK timer replacement using CUDA events
struct StopWatchInterface {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    float elapsed_ms{0.0f};
    bool started{false};
};

inline void sdkCreateTimer(StopWatchInterface** tPtr) {
    *tPtr = new StopWatchInterface();
    cudaEventCreate(&((*tPtr)->start));
    cudaEventCreate(&((*tPtr)->stop));
    (*tPtr)->elapsed_ms = 0.0f;
    (*tPtr)->started = false;
}

inline void sdkStartTimer(StopWatchInterface** tPtr) {
    StopWatchInterface* t = *tPtr;
    t->elapsed_ms = 0.0f;
    t->started = true;
    cudaEventRecord(t->start, 0);
}

inline void sdkStopTimer(StopWatchInterface** tPtr) {
    StopWatchInterface* t = *tPtr;
    if (!t->started) return;
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
    cudaEventElapsedTime(&(t->elapsed_ms), t->start, t->stop);
    t->started = false;
}

inline void sdkResetTimer(StopWatchInterface** tPtr) {
    StopWatchInterface* t = *tPtr;
    t->elapsed_ms = 0.0f;
    t->started = false;
}

inline float sdkGetTimerValue(StopWatchInterface** tPtr) {
    return (*tPtr)->elapsed_ms;
}

inline void sdkDeleteTimer(StopWatchInterface** tPtr) {
    if (!tPtr || !*tPtr) return;
    cudaEventDestroy((*tPtr)->start);
    cudaEventDestroy((*tPtr)->stop);
    delete *tPtr;
    *tPtr = nullptr;
}