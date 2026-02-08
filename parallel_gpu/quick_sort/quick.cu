/*
 * CUDA-Quicksort.h
 *
 * Copyright © 2012-2015 Emanuele Manca
 *
 **********************************************************************************************
 **********************************************************************************************
 *
 	This file is part of CUDA-Quicksort.

    CUDA-Quicksort is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDA-Quicksort is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDA-Quicksort.

    If not, see http://www.gnu.org/licenses/gpl-3.0.txt and http://www.gnu.org/copyleft/gpl.html


  **********************************************************************************************
  **********************************************************************************************
 *
 * Contact: Ing. Emanuele Manca
 *
 * Department of Electrical and Electronic Engineering,
 * University of Cagliari,
 * P.zza D’Armi, 09123, Cagliari, Italy
 *
 * email: emanuele.manca@diee.unica.it
 *
 *
 * This software contains source code provided by NVIDIA Corporation
 * license: http://developer.download.nvidia.com/licenses/general_license.txt
 *
 * this software uses the library of NVIDIA CUDA SDK and the Cederman and Tsigas' GPU Quick Sort
 *
 */

// Build:
// nvcc -O2 -std=c++17 -arch=sm_86 -lineinfo quick_base.cu -o quick_base.exe
//
// Run (single):
// quick_base.exe --size 24 --device 0 --output ../quick_data --reps 1 --seed 12345
//
// Sweep example:
// quick_base.exe --size 24 --device 0 --output ../quick_data --reps 1 --seed 12345 --sweep-threads 128,256,512 --sweep-blocks -1,65536
//
// Notes:
// - Reference-style CUDA Quicksort pipeline in one .cu, scans via thrust inclusive_scan
// - Fixes: integer atomics for uint; CAS-based atomics for double; threadIdx.x typo corrected
// - Safe bounds in globalBitonicSort to prevent shared OOB
// - Preserves your CLI + CSV harness

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define checkCudaErrors(x) gpuErrchk(x)
inline void getLastCudaError(const char* /*msg*/ = "") { gpuErrchk(cudaGetLastError()); }

struct StopWatchInterface { cudaEvent_t start{}, stop{}; };
inline void sdkCreateTimer(StopWatchInterface** t) { *t = (StopWatchInterface*)malloc(sizeof(StopWatchInterface)); cudaEventCreate(&((*t)->start)); cudaEventCreate(&((*t)->stop)); }
inline void sdkResetTimer(StopWatchInterface** /*t*/) {}
inline void sdkStartTimer(StopWatchInterface* t) { cudaEventRecord(t->start, 0); }
inline void sdkStopTimer(StopWatchInterface* t) { cudaEventRecord(t->stop, 0); cudaEventSynchronize(t->stop); }
inline float sdkGetTimerValue(StopWatchInterface* t) { float ms=0; cudaEventElapsedTime(&ms, t->start, t->stop); return ms; }

struct Timings { float h2d_ms=0, kernel_ms=0, d2h_ms=0, total_ms=0; int kernel_invocations=0; };

struct DeviceInfo {
    int deviceId=0; char name[256]{};
    int major=0, minor=0;
    int multiProcessorCount=0;
    int maxThreadsPerBlock=0;
    size_t sharedMemPerBlock=0;
};
static DeviceInfo queryDevice(int dev) {
    cudaDeviceProp p{}; gpuErrchk(cudaGetDeviceProperties(&p, dev));
    DeviceInfo di; di.deviceId = dev;
    snprintf(di.name, sizeof(di.name), "%s", p.name);
    di.major = p.major; di.minor = p.minor;
    di.multiProcessorCount = p.multiProcessorCount;
    di.maxThreadsPerBlock = p.maxThreadsPerBlock;
    di.sharedMemPerBlock = p.sharedMemPerBlock;
    return di;
}

static inline int next_pow2_int(int v) { int p=1; while (p<v) p<<=1; return p; }
std::string timestamp_string() {
    std::time_t t = std::time(nullptr); std::tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64]; std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}
void ensure_dir(const std::filesystem::path &p) {
    std::error_code ec;
    if (!std::filesystem::exists(p)) {
        std::filesystem::create_directories(p, ec);
        if (ec) { std::cerr << "Failed to create directory " << p << ": " << ec.message() << std::endl; exit(1); }
    }
}

// ================== CUDA-Quicksort.h equivalents ==================
typedef unsigned int uint;
#define SHARED_LIMIT 1024
#define GIGA 1073741824

template <typename Type>
struct Block {
    unsigned int begin;
    unsigned int end;
    unsigned int nextbegin;
    unsigned int nextend;
    Type         pivot;
    Type         maxPiv;
    Type         minPiv;
    short        done;
    short        select;
};

template <typename Type>
struct Partition {
    unsigned int ibucket;
    unsigned int from;
    unsigned int end;
    Type         pivot;
};

// ================== scan_common.h equivalents ==================
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

template <typename Type>
inline __device__ void warpScanInclusive2(Type& idata,Type& idata2, volatile Type *s_Data,volatile Type *s_Data2, uint size){
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    s_Data2[pos] = idata2;
    for(uint offset = 1; offset < size; offset <<= 1) {
        s_Data[pos] += s_Data[pos - offset];
        s_Data2[pos] += s_Data2[pos - offset];
    }
    idata = s_Data[pos];
    idata2= s_Data2[pos];
}

template <typename Type>
inline __device__ void warpScanExclusive2(Type& idata,Type& idata2, volatile Type *s_Data,volatile Type *s_Data2, uint size){
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    s_Data2[pos] = idata2;
    for (uint offset = 1; offset < size; offset <<= 1) {
        s_Data[pos] += s_Data[pos - offset];
        s_Data2[pos] += s_Data2[pos - offset];
    }
    idata  = s_Data[pos] - idata;
    idata2 = s_Data2[pos] - idata2;
}

template <typename Type>
inline __device__ void scan1Inclusive2(Type& idata,Type& idata2, volatile Type *s_Data, uint size){
    volatile Type* s_Data2;
    s_Data2 = s_Data + blockDim.x*2;

    if(size > WARP_SIZE){
        warpScanInclusive2(idata,idata2, s_Data,s_Data2, WARP_SIZE);
        __syncthreads();
        if( (threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1) ) {
            s_Data[threadIdx.x >> LOG2_WARP_SIZE]  = idata;
            s_Data2[threadIdx.x >> LOG2_WARP_SIZE] = idata2;
        }
        __syncthreads();
        if( threadIdx.x < (blockDim.x / WARP_SIZE)){
            Type val  = s_Data[threadIdx.x];
            Type val2 = s_Data2[threadIdx.x];
            warpScanExclusive2(val,val2, s_Data,s_Data2, size >> LOG2_WARP_SIZE);
            s_Data[threadIdx.x]  = val;
            s_Data2[threadIdx.x] = val2;
        }
        __syncthreads();
        idata  += s_Data[threadIdx.x >> LOG2_WARP_SIZE];
        idata2 += s_Data2[threadIdx.x >> LOG2_WARP_SIZE];
    } else {
        warpScanInclusive2(idata,idata2, s_Data,s_Data2, size);
    }
}

template <typename Type>
inline __device__ void warpCompareInclusive(Type& idata,Type& idata2, volatile Type *s_Data, uint size){
    volatile Type* s_Data2;
    s_Data2 = s_Data + blockDim.x * 2;
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1)); // fixed
    s_Data[pos]  = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos]  = idata;
    s_Data2[pos] = idata2;
    for (uint offset = 1; offset < size; offset <<= 1) {
        s_Data[pos]  = max(s_Data[pos],  s_Data[pos - offset]);
        s_Data2[pos] = min(s_Data2[pos], s_Data2[pos - offset]);
    }
    idata  = s_Data[pos];
    idata2 = s_Data2[pos];
}

template <typename Type>
inline __device__ void compareInclusive(Type& idata,Type& idata2, volatile Type *s_Data, uint size){ // idata=max, idata2=min
    volatile Type* s_Data2;
    s_Data2 = s_Data + blockDim.x*2;
    warpCompareInclusive(idata,idata2, s_Data, WARP_SIZE);
    __syncthreads();
    if( (threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1) ) {
        s_Data[threadIdx.x >> LOG2_WARP_SIZE]  = idata;
        s_Data2[threadIdx.x >> LOG2_WARP_SIZE] = idata2;
    }
    __syncthreads();
    if( threadIdx.x < (blockDim.x /WARP_SIZE)) {
        Type val  = s_Data[threadIdx.x];
        Type val2 = s_Data2[threadIdx.x];
        warpCompareInclusive(val,val2, s_Data, size >> LOG2_WARP_SIZE);
        s_Data[threadIdx.x]  = val;
        s_Data2[threadIdx.x] = val2;
    }
    __syncthreads();
    idata  = max(idata,  s_Data[threadIdx.x >> LOG2_WARP_SIZE]) ;
    idata2 = min(idata2, s_Data2[threadIdx.x >> LOG2_WARP_SIZE]);
}

// thrust-backed scans (batchSize=1)
extern "C" size_t scanInclusiveShort(uint *d_Dst, uint *d_Src, uint batchSize, uint arrayLength) {
    if (batchSize != 1) return 0;
    thrust::device_ptr<uint> src(d_Src), dst(d_Dst);
    thrust::inclusive_scan(src, src + arrayLength, dst);
    return arrayLength * sizeof(uint);
}
extern "C" size_t scanInclusiveLarge(uint *d_Dst, uint *d_Src, uint batchSize, uint arrayLength) {
    return scanInclusiveShort(d_Dst, d_Src, batchSize, arrayLength);
}
inline void initScan() {}
inline void closeScan() {}

// ================== Double CAS atomics + type-dispatch ==================
__device__ inline double atomicMax_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int assumed;
    unsigned long long int old = *address_as_ull;
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(max(val ,__longlong_as_double(assumed))));
    while (assumed != old)
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(max(val ,__longlong_as_double(assumed))));
    }
    return __longlong_as_double(old);
}
__device__ inline double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(min(val ,__longlong_as_double(assumed))));
    while (assumed != old)
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(min(val ,__longlong_as_double(assumed))));
    }
    return __longlong_as_double(old);
}

template <typename T>
__device__ inline void atomicUpdateMaxMin(T* maxPiv, T* minPiv, T rmax, T lmin) {
    atomicMax(maxPiv, rmax);
    atomicMin(minPiv, lmin);
}
template <>
__device__ inline void atomicUpdateMaxMin<double>(double* maxPiv, double* minPiv, double rmax, double lmin) {
    atomicMax_double(maxPiv, rmax);
    atomicMin_double(minPiv, lmin);
}

// ================== bfind helper + Comparator ==================
static __device__ __forceinline__ unsigned int __qsflo(unsigned int word)
{
    unsigned int ret;
    asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
}
template <typename Type>
__device__ inline void Comparator(Type& valA, Type& valB, uint dir){
    Type t;
    if( (valA > valB) == dir ){
        t = valA; valA = valB; valB = t;
    }
}

// ================== Kernels ==================
template <typename Type>
__global__ void globalBitonicSort(Type* indata,Type*outdata, Block<Type>* bucket, bool inputSelect)
{
    __shared__ uint shared[1024];

    Type* data;
    Block<Type> cord = bucket[blockIdx.x];

    uint size=cord.end-cord.begin;
    bool select = !(cord.select);

    if(size>1024 || size==0) return;

    unsigned int bitonicSize = 1 << (__qsflo(size-1U)+1);

    data = select ? indata : outdata;

    for(int i=threadIdx.x;i<(int)size;i+=blockDim.x)
         shared[i] = data[i+cord.begin];

    for(int i=threadIdx.x+size;i<(int)bitonicSize;i+=blockDim.x)
        shared[i] = 0xffffffff;

    __syncthreads();

    for(uint sz = 2; sz < bitonicSize; sz <<= 1){
        uint ddd = 1 ^ ( (threadIdx.x & (sz / 2)) != 0 );
        for(uint stride = sz / 2; stride > 0; stride >>= 1){
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            // safe bounds to avoid shared OOB
            if (pos + stride < bitonicSize) {
                Comparator(shared[pos + 0], shared[pos + stride], ddd);
            }
        }
    }

    for(uint stride = bitonicSize / 2; stride > 0; stride >>= 1){
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        if (pos + stride < bitonicSize) {
            Comparator(shared[pos + 0], shared[pos + stride], 1);
        }
    }

    __syncthreads();
    for(int i=threadIdx.x;i<(int)size;i+=blockDim.x)
        indata[i+cord.begin] = shared[i];
}

template <typename Type>
__global__ void quick(Type* indata,Type* buffer,  Partition<Type>* partition, Block<Type>* bucket )
{
    __shared__ Type sh_out[1024];

    __shared__ uint start1,end1;
    __shared__ uint left,right;

    int tix = threadIdx.x;

    uint start  = partition[blockIdx.x].from;
    uint end    = partition[blockIdx.x].end;
    Type pivot  = partition[blockIdx.x].pivot;
    uint nseq   = partition[blockIdx.x].ibucket;

    uint lo=0;
    uint hi=0;

    Type lmin = (Type)0xffffffff;
    Type rmax = (Type)0;

    Type d;

    if(tix+start<end)
    {
        d = indata[tix+start];
        lo=(d<pivot)*(lo+1)+(d>=pivot)*lo;
        hi=(d<=pivot)*(hi)+(d>pivot)*(hi+1);
        lmin = d;
        rmax = d;
    }

    for(uint i=tix+start+blockDim.x;i<end;i+=blockDim.x)
    {
        Type dv= indata[i];
        lo = ( dv <  pivot ) *(lo+1) + ( dv >= pivot )*lo;
        hi = ( dv <= pivot ) *(hi)   +  (dv >  pivot )*(hi+1);
        lmin = min(lmin,dv);
        rmax = max(rmax,dv);
    }

    // idata=max, idata2=min according to reference
    compareInclusive(rmax,lmin,(Type*) sh_out, blockDim.x);
    __syncthreads();

    if(tix==blockDim.x-1)
    {
        atomicUpdateMaxMin(&bucket[nseq].maxPiv, &bucket[nseq].minPiv, rmax, lmin);
    }

    __syncthreads();

    scan1Inclusive2(lo,hi,(uint*) sh_out, blockDim.x);
    lo = lo-1;
    hi = SHARED_LIMIT-hi;

    if(tix==blockDim.x-1)
    {
        left  = lo+1;
        right = SHARED_LIMIT-hi;

        start1 = atomicAdd(&bucket[nseq].nextbegin,left);
        end1   = atomicSub(&bucket[nseq].nextend, right);
    }

    __syncthreads();

    if(tix+start<end)
    {
        if(d<pivot) { sh_out[lo]=d; lo--; }
        if(d>pivot) { sh_out[hi]=d; hi++; }
    }

    for(uint i=start+tix+blockDim.x;i<end;i+=blockDim.x)
    {
        Type dv=indata[i];
        if(dv<pivot) { sh_out[lo]=dv; lo--; }
        if(dv>pivot) { sh_out[hi]=dv; hi++; }
    }

    __syncthreads();

    for(uint i=tix ;i<SHARED_LIMIT;i+=blockDim.x)
    {
        if(i<left)
            buffer[start1+i]=sh_out[i];

        if(i>=SHARED_LIMIT-right)
            buffer[end1+i-SHARED_LIMIT]=sh_out[i];
    }
}

template <typename Type>
__global__ void partitionAssign(struct Block<Type>* bucket,uint* npartitions,struct Partition<Type>* partition)
{
    int tx=threadIdx.x;
    int bx=blockIdx.x;

    uint beg   = bucket[bx].nextbegin;
    uint end   = bucket[bx].nextend;
    Type pivot = bucket[bx].pivot;

    uint from;
    uint to;

    if(bx>0) { from=npartitions[bx-1]; to=npartitions[bx]; }
    else     { from=0;                 to=npartitions[bx]; }

    uint i=tx+from;

    if(i<to )
    {
        uint begin=beg+SHARED_LIMIT*tx;
        partition[i].from=begin;
        partition[i].end=begin+SHARED_LIMIT;
        partition[i].pivot=pivot;
        partition[i].ibucket=bx;
    }

    for(uint j=tx+from+blockDim.x;j<to ;j+=blockDim.x)
    {
        uint begin=beg+SHARED_LIMIT*(j-from);
        partition[j].from=begin;
        partition[j].end=begin+SHARED_LIMIT;
        partition[j].pivot=pivot;
        partition[j].ibucket=bx;
    }
    __syncthreads();
    if(tx==0 && to-from>0) partition[to-1].end=end;
}

template <typename Type>
__global__ void insertPivot(Type* data,struct Block<Type>* bucket,int nbucket)
{
    Type pivot      = bucket[blockIdx.x].pivot;
    uint start      = bucket[blockIdx.x].nextbegin;
    uint end        = bucket[blockIdx.x].nextend;
    bool is_altered = bucket[blockIdx.x].done;

    if(is_altered && (int)blockIdx.x<nbucket)
        for(uint j=start+threadIdx.x; j<end; j+=blockDim.x)
            data[j]=pivot;
}

template <typename Type>
__global__ void bucketAssign(Block<Type>* bucket,uint*npartitions,int nbucket,int select)
{
    uint i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i<(uint)nbucket){
            bool is_altered=bucket[i].done;
            if(is_altered )
            {
                uint orgbeg = bucket[i].begin;
                uint from    = bucket[i].nextbegin;
                uint orgend = bucket[i].end;
                uint end    = bucket[i].nextend;
                Type pivot  = bucket[i].pivot;
                Type minPiv = bucket[i].minPiv;
                Type maxPiv = bucket[i].maxPiv;

                Type lmaxpiv = min(pivot,maxPiv);
                Type rminpiv = max(pivot,minPiv);

                bucket[i+nbucket].begin = orgbeg;
                bucket[i+nbucket].nextbegin   = orgbeg;
                bucket[i+nbucket].nextend    = from;
                bucket[i+nbucket].end = from;
                bucket[i+nbucket].pivot  = (minPiv+lmaxpiv)/2;

                bucket[i+nbucket].done   = (from-orgbeg)>1024 && (minPiv!=maxPiv);
                bucket[i+nbucket].select=select;
                bucket[i+nbucket].minPiv = (Type)0xffffffff;
                bucket[i+nbucket].maxPiv = (Type)0x0;

                if(!bucket[i+nbucket].done)
                     npartitions[i+nbucket] = 0;
                else npartitions[i+nbucket] = (from-orgbeg+SHARED_LIMIT-1)/SHARED_LIMIT;

                bucket[i].begin = end;
                bucket[i].nextbegin   = end;
                bucket[i].nextend    = orgend;
                bucket[i].pivot  = (rminpiv+maxPiv)/2+1;

                bucket[i].done   = (orgend-end)>1024 && (minPiv!=maxPiv);
                bucket[i].select=select;
                bucket[i].minPiv = (Type)0xffffffff;
                bucket[i].maxPiv = (Type)0x0;

                if(!bucket[i].done)
                    npartitions[i]=0;
                else
                    npartitions[i]=(orgend-end+SHARED_LIMIT-1)/SHARED_LIMIT;
            }
        }
}

template <typename Type>
__global__ void init(Type* data,Block<Type>* bucket,uint* npartitions,int size, int nblocks)
{
    uint i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i<(uint)nblocks)
    {
        bucket[i].nextbegin   = 0;
        bucket[i].begin = 0;

        bucket[i].nextend    = 0 + size*(i==0);
        bucket[i].end = 0 + size*(i==0);
        npartitions[i]   = 0;
        bucket[i].done   = (short)(false + (i==0));
        bucket[i].select   = (short)false;
        bucket[i].maxPiv = (Type)0x0;
        bucket[i].minPiv = (Type)0xffffffff;
        bucket[i].pivot = data[size/2];
    }
}

// ================== Sort driver ==================
template <typename Type>
void sort(Type* inputData,Type* outputData, uint size,uint threadCount,int device, double* wallClock)
{
    cudaSetDevice(device);
    cudaGetLastError();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    StopWatchInterface* htimer=nullptr;
    Type* ddata;
    Type* dbuffer;

    Block<Type>* dbucket;
    struct Partition<Type>* partition;
    uint* npartitions1,*npartitions2;

    uint*cudaBlocks=(uint*)malloc(4);

    uint blocks = (size + SHARED_LIMIT-1)/SHARED_LIMIT;
    uint nblock=10*blocks;
    int partition_max= 262144;

    unsigned long long int total = (unsigned long long)partition_max*sizeof(Block<Type>) + (unsigned long long)nblock*sizeof(Partition<Type>) + 2ull*partition_max*sizeof(uint) +3ull*(size)*sizeof(Type);

    printf("\nINFO: Device Memory consumed is %.3f GB out of %.3f GB of available memory\n", ((double)total/GIGA), (double)deviceProp.totalGlobalMem/GIGA);

    sdkCreateTimer(&htimer);
    checkCudaErrors( cudaMalloc  ((void**)&dbucket   , partition_max*sizeof(Block<Type>)) );
    checkCudaErrors( cudaMalloc  ((void**)&partition , nblock*sizeof(Partition<Type>)) );

    checkCudaErrors(cudaMalloc((void**)&npartitions1,partition_max*sizeof(uint)) );
    checkCudaErrors(cudaMalloc((void**)&npartitions2,partition_max*sizeof(uint)) );

    checkCudaErrors(cudaMalloc((void**)&dbuffer,(size)*sizeof(Type)));
    checkCudaErrors(cudaMalloc((void**)&ddata  ,(size)*sizeof(Type)));

    checkCudaErrors(cudaMemcpy(ddata, inputData, size*sizeof(Type), cudaMemcpyHostToDevice) );

    initScan();

    cudaFuncSetCacheConfig(init<Type>,                cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(insertPivot<Type>,         cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(bucketAssign<Type>,        cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(partitionAssign<Type>,     cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(quick<Type>,               cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(globalBitonicSort<Type>,   cudaFuncCachePreferShared);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&htimer);
    sdkStartTimer(htimer);

    init<Type><<<(nblock+255)/256,256>>>(ddata,dbucket,npartitions1,(int)size,partition_max);

    uint nbucket     = 1;
    uint numIterations  = 0;
    bool inputSelect = true;

    *cudaBlocks=blocks;
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("init() execution FAILED\n");
    checkCudaErrors( cudaMemcpy(&npartitions2[0], cudaBlocks,  sizeof(uint), cudaMemcpyHostToDevice) );

    while(1)
    {
        if(numIterations>0)
        {
            if(nbucket<=1024)
                scanInclusiveShort(npartitions2, npartitions1, 1, nbucket);
            else
                scanInclusiveLarge(npartitions2, npartitions1, 1, nbucket);

            checkCudaErrors( cudaMemcpy(cudaBlocks, &npartitions2[nbucket-1],  sizeof(uint), cudaMemcpyDeviceToHost) );
        }

        if(*cudaBlocks==0)
            break;

        partitionAssign<Type><<<nbucket,1024>>>(dbucket,npartitions2,partition);
        cudaDeviceSynchronize();
        getLastCudaError("partitionAssign() execution FAILED\n");

        if(inputSelect)
            quick<Type><<<*cudaBlocks,threadCount>>>(ddata,dbuffer,partition,dbucket);
        else
            quick<Type><<<*cudaBlocks,threadCount>>>(dbuffer,ddata,partition,dbucket);
        cudaDeviceSynchronize();
        getLastCudaError("quick() execution FAILED\n");

        insertPivot<Type><<<nbucket,512>>>(ddata,dbucket,nbucket);

        bucketAssign<Type><<<(nbucket+255)/256,256>>>(dbucket,npartitions1,nbucket,inputSelect);
        cudaDeviceSynchronize();
        getLastCudaError("insertPivot() or bucketAssign() execution FAILED\n");

        nbucket*=2;

        inputSelect = !inputSelect;
        numIterations++;
        if(nbucket>deviceProp.maxGridSize[0])
            break;
    }

    printf("Iteracija %u\n", numIterations);
    if(nbucket>deviceProp.maxGridSize[0])
        fprintf(stderr, "ERROR: CUDA-Quicksort can't terminate sorting as the block threads needed to finish it are more than the Maximum x-dimension\n");
    else
        globalBitonicSort<Type><<<nbucket,512,0>>>(ddata,dbuffer,dbucket,inputSelect);

    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("globalBitonicSort() execution FAILED\n");

    sdkStopTimer(htimer);
    *wallClock=sdkGetTimerValue(htimer);

    checkCudaErrors(cudaMemcpy(outputData, ddata, size*sizeof(Type), cudaMemcpyDeviceToHost) );

    checkCudaErrors( cudaFree(ddata) );
    checkCudaErrors( cudaFree(dbuffer) );
    checkCudaErrors( cudaFree(dbucket) );
    checkCudaErrors( cudaFree(npartitions2));
    checkCudaErrors( cudaFree(npartitions1));
    free(cudaBlocks);

    closeScan();
    return ;
}

// ================== Public wrappers ==================
extern "C"
void CUDA_Quicksort(uint* inputData, uint* outputData, uint dataSize, uint threadCount, int Device, double* wallClock)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);

    if(deviceProp.major<2)
    {
        fprintf(stderr, "Error: the GPU device %d has a Compute Capability of %d.%d, while a Compute Capability of 2.x is required to run the code\n",
                Device, deviceProp.major, deviceProp.minor);

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        fprintf(stderr, "       the Host system has the following GPU devices:\n");

        for (int device = 0; device < deviceCount; device++) {
            fprintf(stderr, "\t  the GPU device %d has Compute Capability %d.%d\n",
                    device, deviceProp.major, deviceProp.minor);
        }

        return;
    }

    sort<uint>(inputData,outputData, dataSize,threadCount,Device, wallClock);
}

extern "C"
void CUDA_Quicksort_64(double* inputData,double* outputData, uint dataSize, uint threadCount, int Device, double* wallClock)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);

    if(deviceProp.major<2)
    {
        fprintf(stderr, "Error: the GPU device %d has a Compute Capability of %d.%d, while a Compute Capability of 2.x is required to run the code\n",
                Device, deviceProp.major, deviceProp.minor);

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        fprintf(stderr, "       the Host system has the following GPU devices:\n");

        for (int device = 0; device < deviceCount; device++) {
            fprintf(stderr, "\t  the GPU device %d has Compute Capability %d.%d\n",
                    device, deviceProp.major, deviceProp.minor);
        }

        return;
    }

    sort<double>(inputData,outputData, dataSize,threadCount,Device,wallClock);
}

// ================== CSV harness ==================
static std::vector<int> parse_list(const std::string &s) {
    std::vector<int> out; std::stringstream ss(s); std::string token;
    while (std::getline(ss, token, ',')) { try { out.push_back(std::stoi(token)); } catch (...) {} }
    return out;
}
struct CLI {
    int sizePow2 = 24;
    int device = 0;
    int reps = 1;
    unsigned int seed = 12345u;
    std::string output_dir = "../quick_data";
    std::string sweep_name;
    std::vector<int> sweep_threads;
    std::vector<int> sweep_blocks;
};
static CLI parse_cli(int argc, char** argv) {
    CLI o;
    for (int i=1;i<argc;i++) {
        std::string s(argv[i]);
        if (s=="--device" && i+1<argc) o.device = std::stoi(argv[++i]);
        else if (s=="--size" && i+1<argc) o.sizePow2 = std::stoi(argv[++i]);
        else if (s=="--reps" && i+1<argc) o.reps = std::stoi(argv[++i]);
        else if ((s=="--output" || s=="--output-dir") && i+1<argc) o.output_dir = argv[++i];
        else if (s=="--seed" && i+1<argc) o.seed = (unsigned int)std::stoul(argv[++i]);
        else if (s=="--sweep-name" && i+1<argc) o.sweep_name = argv[++i];
        else if (s=="--sweep-threads" && i+1<argc) o.sweep_threads = parse_list(argv[++i]);
        else if (s=="--sweep-blocks" && i+1<argc) o.sweep_blocks  = parse_list(argv[++i]);
    }
    return o;
}

int main(int argc, char** argv) {
    CLI opts = parse_cli(argc, argv);
    size_t N = (opts.sizePow2 >= 1 && opts.sizePow2 <= 31) ? (size_t)1 << opts.sizePow2 : (size_t)1 << 24;

    int devCount = 0; gpuErrchk(cudaGetDeviceCount(&devCount));
    if (devCount == 0) { std::cerr << "No CUDA devices found.\n"; return 1; }
    if (opts.device < 0 || opts.device >= devCount) {
        std::cerr << "Invalid device id: " << opts.device << ", devices available: " << devCount << std::endl;
        return 1;
    }
    gpuErrchk(cudaSetDevice(opts.device));
    DeviceInfo di = queryDevice(opts.device);
    printf("Device %d: %s, CC=%d.%d, SMs=%d, maxThreadsPerBlock=%d, sharedMemPerBlock=%zu\n",
           di.deviceId, di.name, di.major, di.minor, di.multiProcessorCount, di.maxThreadsPerBlock, di.sharedMemPerBlock);

    ensure_dir(opts.output_dir);
    std::string csv_name = opts.sweep_name.empty() ? ("quick_base_sweep_" + timestamp_string() + ".csv") : opts.sweep_name;
    std::filesystem::path csv_path = std::filesystem::path(opts.output_dir) / csv_name;
    std::ofstream csv(csv_path);
    if (!csv.is_open()) { std::cerr << "Failed to open CSV: " << csv_path << std::endl; return 1; }

    csv << "timestamp,run_id,size,next_pow2,rep,seed,device_id,device_name,compute_capability,sm_count,threads,blocks,algorithm,"
           "h2d_ms,total_kernel_ms,kernel_invocations,d2h_ms,total_ms,throughput_Melems_s,verify_passed\n";
    csv << std::fixed << std::setprecision(3);

    std::vector<int> threads_list = opts.sweep_threads.empty() ? std::vector<int>{256} : opts.sweep_threads;
    std::vector<int> blocks_list  = opts.sweep_blocks.empty()  ? std::vector<int>{-1} : opts.sweep_blocks;

    int run_id = 0;
    for (int th : threads_list) {
        int threadCount = th; if (threadCount > di.maxThreadsPerBlock) threadCount = di.maxThreadsPerBlock;

        for (int bl : blocks_list) {
            int blocks_override = bl;
            std::cout << "=== Sweep: threads=" << threadCount
                      << " blocks_override=" << blocks_override
                      << " size=" << (uint64_t)N << " (2^" << opts.sizePow2 << ") ===\n";

            for (int rep = 0; rep < opts.reps; ++rep) {
                ++run_id;

                std::vector<int> h_in(N);
                std::mt19937 rng(opts.seed + run_id + rep);
                std::uniform_int_distribution<int> dist(INT32_MIN/2, INT32_MAX/2);
                for (size_t i = 0; i < N; i++) h_in[i] = dist(rng);

                std::vector<uint> u_in(N), u_out(N);
                for (size_t i=0;i<N;++i) u_in[i] = (uint)(h_in[i] ^ 0x80000000);

                double wallClockMs = 0.0;
                CUDA_Quicksort(u_in.data(), u_out.data(), (uint)N, (uint)threadCount, opts.device, &wallClockMs);

                std::vector<int> h_out(N);
                for (size_t i=0;i<N;++i) h_out[i] = (int)(u_out[i] ^ 0x80000000);

                Timings T{};
                T.kernel_ms = (float)wallClockMs; T.total_ms = (float)wallClockMs;

                bool ok = std::is_sorted(h_out.begin(), h_out.end());

                int natural_blocks = (int)((N + threadCount - 1) / threadCount);
                int blocks_used = (blocks_override > 0) ? blocks_override : natural_blocks;
                double throughput = (double)N / (T.total_ms / 1000.0) / 1e6;
                std::string devname(di.name); for (auto &c : devname) if (c == ',') c = ';';

                csv << timestamp_string() << "," << run_id << "," << (uint64_t)N << "," << next_pow2_int((int)N) << ","
                    << rep << "," << opts.seed << "," << di.deviceId << ","
                    << "\"" << devname << "\"" << "," << di.major << "." << di.minor << ","
                    << di.multiProcessorCount << "," << threadCount << "," << blocks_used << ","
                    << "\"CUDA-Quicksort (from scratch)\"" << ","
                    << T.h2d_ms << "," << T.kernel_ms << "," << T.kernel_invocations << ","
                    << T.d2h_ms << "," << T.total_ms << "," << throughput << "," << (ok ? "1" : "0") << "\n";
                csv.flush();

                std::cout << "run=" << run_id << " OK=" << (ok ? 1 : 0)
                          << " kernel=" << T.kernel_ms << "ms total=" << T.total_ms
                          << "ms throughput=" << std::setprecision(3) << throughput << " Me/s"
                          << std::setprecision(3) << std::endl;
            }
        }
    }

    csv.close();
    std::cout << "CSV results written to " << (std::filesystem::absolute(csv_path)).string() << std::endl;
    return 0;
}