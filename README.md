# Sorting Algorithms Parallelization

## Project Purpose

This project compares **multiple sorting algorithms** implemented in two stages:

1. **Sequential (baseline)** – standard single-threaded C++ implementations  
2. **Parallel CPU** – optimized implementations using multi-core and SIMD parallelism  
3. *(Optional later stage)* **Parallel GPU** – GPU acceleration using CUDA, SYCL, or OpenCL (not yet implemented)

The goal is to **benchmark execution times**, **visualize speedups**, and analyze how various algorithms scale with CPU-level parallelism across different hardware configurations.

---

## Folder Structure

```
.
├── README.md                    # Developer guide
├── data                         # Input arrays and benchmark outputs (CSV or JSON)
├── sequential                   # Sequential sorting algorithm implementations
├── parallel_cpu                 # CPU-parallel sorting implementations
├── parallel_gpu                 # Placeholder for GPU-parallel implementations (future work)
├── plots                        # Scripts / data for visualization
├── include                      # Shared headers and utilities
├── build                        # Compiled output files
├── run_project.sh               # Bash script to run all executables, save results
├── run_project.ps1              # PowerShell equivalent for Windows
├── run_plots_generating.ps1     # PowerShell script for generating plots
└── CMakeLists.txt               # Project build configuration

```

---

## Dependencies and Environment Setup

### Required Tools

- **C++17 or newer**
- **CMake ≥ 3.16**
- **Compiler**:
  - Clang or GCC on Linux/macOS  
  - MSVC on Windows  
- **Optional CPU-parallelization libraries:**
  - [OpenMP](https://www.openmp.org/) – for multi-threading across cores  
  - [Intel TBB (oneTBB)](https://github.com/oneapi-src/oneTBB) – for task-based parallelism  
  - [SIMD intrinsics or std::experimental::simd](https://en.cppreference.com/w/cpp/experimental/simd) – for vectorized operations
  > If `std::experimental::simd` is unavailable, consider using compiler intrinsics (`<immintrin.h>`) for vectorization.


---

# Environment Setup

## On macOS / Linux

> **Note (macOS M1/M2):** Apple Clang lacks OpenMP support. Install Homebrew LLVM or GCC to enable OpenMP:
> 
> ```bash
> brew install libomp gcc
> ```


```bash
# Clone the repository
git clone https://github.com/goralija/Sorting-algorithms-parallelization.git
cd Sorting-algorithms-parallelization

# Create and enter build directory
mkdir build && cd build

# Configure and build with CMake (enabling OpenMP)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ ..
make -j$(nproc)
```

## On Windows (PowerShell)

### 1. Install MSYS2

1. Download and install MSYS2: [https://www.msys2.org/](https://www.msys2.org/)
2. Open **MSYS2 MSYS** terminal and update packages:

```bash
pacman -Syu
```

3. Close the terminal, reopen **MSYS2 MSYS**, and install required packages:

```bash
pacman -S --needed base-devel mingw-w64-x86_64-toolchain mingw-w64-x86_64-clang mingw-w64-x86_64-ninja mingw-w64-x86_64-cmake mingw-w64-x86_64-python
```

> This provides gcc, g++, cmake, make, and Python.

---

### 2. Open MinGW64 Shell

- Launch **MSYS2 MinGW 64-bit** shell from Start Menu.
- This ensures MinGW64 compiler and tools are available in PATH.

---

### 3. Configure VS Code Terminal

1. Open VS Code → Settings → Terminal → Integrated → Profiles
2. Add a new profile for MinGW64:

```json
"MSYS2 MinGW64": {
    "path": "C:\\msys64\\mingw64.exe"
}
```

3. Open a terminal in VS Code using **MSYS2 MinGW64**.

---

### 4. Clone and Run Project

```bash
git clone https://github.com/goralija/Sorting-algorithms-parallelization.git
cd Sorting-algorithms-parallelization
powershell ./run_project.ps1
```

- This will:
  1. Build all C++ executables using MinGW64 + CMake
  2. Run benchmarks for sequential and parallel CPU algorithms
  3. Save results in `data/benchmark.csv`
  4. Automatically generate plots in `plots/` (requires Python 3.10+)

> Always use **MSYS2 MinGW64 shell** for building and running; do not run CMake directly in PowerShell.

---

### 5. Python Environment

- `run_project.ps1` automatically creates a Python virtual environment `venv/` if needed.
- Required packages: `matplotlib`, `pandas`, `numpy`
- No manual activation is required when using `run_project.ps1`.

#### Manual Activation (optional)
```powershell
.venv\Scripts\Activate.ps1
python plots/plot_results.py
```

---

> All compiled binaries will appear in the `build/` folder. Use `rm -rf build` to clean and rebuild.

---

## Running Project Pipeline

Instead of running each binary manually, you can use the provided scripts:

### macOS / Linux
```bash
bash run_project.sh
```

### Windows (MSYS MINGW64 SHELL)
```
powershell .\run_project.ps1
```
> Always use **MSYS2 MinGW64 shell** for building and running; do not run CMake directly in PowerShell.

These scripts:

- Are still in development, and you can upgrade them if you find bug
- Rebuild executables
- Run all sequential and parallel CPU executables automatically  
- Accept array size arguments if implemented in C++ (`argv[1]`)  
- Save execution times to `data/benchmark.csv` for plotting
- Plot results

---

## Development Workflow

1. **Add new sorting algorithm**
   - Create a new `.cpp` file under `sequential/` or `parallel_cpu/`
   - Implement `sort(std::vector<int>& arr)` function
   - Ensure results are validated (compare with `std::sort()` output)

2. **Benchmark your algorithm**
   - Use `std::chrono::high_resolution_clock` to measure runtime
   - Save results in CSV format:
     ```
     algorithm,mode,array_size,time_ms,threads
     ```

3. **Parallelize gradually**
   - Start from sequential baseline
   - Add thread/task parallelism (OpenMP or threads)
   - Add SIMD optimizations where applicable

---

## Notes / Best Practices

- **Dynamic executables:** All executables are automatically named with prefixes `sequential_` or `parallel_cpu_` to avoid name conflicts.  
- **SIMD:** Use `std::experimental::simd` or intrinsics for vectorized operations.  
- **Thread management:** Avoid excessive thread creation in recursive algorithms.  
- **Data size:** Use ≥ 10⁵ elements for CPU benchmarks.  
- **Reproducibility:** Repeat tests multiple times and average results.  
- **Cross-platform:** Code should compile under Clang, GCC, and MSVC with C++17 enabled.

---

## Benchmarking and Plotting

### Prerequisites

- **Python 3.10 or newer**
- Required Python libraries: `matplotlib`, `pandas`, `numpy`

### Setting Up Python Environment

#### On macOS / Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### On Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running Benchmarks and Generating Plots

The `run_project` scripts automatically:
1. Build and run all sorting algorithm executables
2. Save benchmark results to `data/benchmark.csv`
3. Set up Python virtual environment (if needed)
4. Generate visualization plots in `plots/` directory

### Manual Plot Generation

If you want to generate plots manually from existing benchmark data:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Run plotting script
python plots/plot_results.py
```

---

## Future Work (GPU Parallelization)

- GPU acceleration (CUDA, SYCL, or OpenCL) will be implemented in the next phase.
- Planned algorithms:
  - GPU-optimized **Bitonic Sort**
  - GPU **Radix Sort**
  - Compare performance against CPU-parallelized equivalents

