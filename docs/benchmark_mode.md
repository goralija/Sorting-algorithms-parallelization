# Benchmark Measurement Mode

## Overview

The project now supports a **benchmark measurement mode** designed specifically for profiling with tools like Intel VTune. This mode minimizes overhead by removing unnecessary operations that could bias performance measurements.

## Configuration

Add the following to your `config.yaml`:

```yaml
benchmark_measurement: true   # Enable pure benchmark mode
```

## Behavior

### When `benchmark_measurement: true`
- ✅ Only the sorting algorithm executes (pure performance measurement)
- ❌ No timing measurements (avoids timer overhead)
- ❌ No array verification (`std::is_sorted` check skipped)
- ❌ No console output (avoids I/O overhead)
- ❌ No CSV writing (avoids file I/O overhead)
- ❌ No array printing (regardless of `print_array` setting)

**Use case**: Profiling with VTune, perf, or other performance analysis tools

### When `benchmark_measurement: false` (default)
- ✅ Full verification with `std::is_sorted`
- ✅ Timing measurements with `Timer` class
- ✅ Console output with algorithm details
- ✅ CSV benchmark results
- ✅ Optional array printing (if `print_array: true`)
- ✅ Error logging for failed sorts

**Use case**: Normal benchmarking, debugging, and result collection

## Command Line Usage

The `--benchmark` (or `-b`) flag is automatically passed by the shell scripts when `benchmark_measurement: true` is set in config.yaml.

Manual usage:
```bash
./build/sequential_naive_quick_sort 8192 random 42 --benchmark
```

Or with array printing disabled:
```bash
./build/sequential_naive_quick_sort 8192 random 42 --print-array --benchmark
```

## Technical Details

### Modified Files

1. **`include/main_template.hpp`**
   - Added `benchmark_mode` parameter parsing
   - Early return after sorting when in benchmark mode
   - Skips all verification and I/O operations

2. **`run_project.ps1`** (PowerShell)
   - Reads `benchmark_measurement` from config.yaml
   - Passes `--benchmark` flag to executables when enabled
   - Skips output parsing in benchmark mode

3. **`run_project.sh`** (Bash)
   - Reads `benchmark_measurement` from config.yaml using `yq`
   - Passes `--benchmark` flag to executables when enabled
   - Skips output parsing in benchmark mode

## Example Workflow

### For VTune Profiling
```yaml
# config.yaml
sizes: [8192]
types: [random]
seed: 42
print_array: false
benchmark_measurement: true  # Enable pure benchmark mode
```

Run with VTune:
```bash
# Build and run once to generate executables
./run_project.sh

# Profile specific algorithm
vtune -collect hotspots ./build/sequential_naive_quick_sort 8192 random 42 --benchmark
```

### For Normal Benchmarking
```yaml
# config.yaml
sizes: [8192, 16384, 32768]
types: [random, sorted, reversed]
seed: 42
print_array: false
benchmark_measurement: false  # Full verification mode
```

```bash
./run_project.sh  # Generates benchmark.csv with timing results
```

## Benefits

1. **Unbiased Measurements**: Removes timing, verification, and I/O overhead
2. **VTune Compatibility**: Clean profiling data focused only on sorting logic
3. **Flexibility**: Easy toggle between profiling and benchmarking modes
4. **Safety**: Default mode still includes full verification
