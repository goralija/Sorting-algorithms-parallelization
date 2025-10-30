# Still in development - modify as needed

#!/bin/bash
# run_executables.sh
# Rebuilds all executables, runs them with different input sizes, and logs results

BUILD_DIR="build"
DATA_DIR="data"
SIZES=(5000000 50000000 500000000)
SIZES_FINAL=(10000 100000 500000 1000000 5000000 10000000 50000000 100000000 500000000 1000000000)

# Clean and rebuild
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ ..
make -j$(nproc)
cd ..

# Prepare output folder and CSV
mkdir -p ${DATA_DIR}
mv ${DATA_DIR}/benchmark.csv ${DATA_DIR}/benchmark_backup_$(date +%s).csv 2>/dev/null || true
OUTFILE="${DATA_DIR}/benchmark.csv"
echo "Algorithm,ArraySize,TimeMs" > "${OUTFILE}"

# Run all executables with various input sizes
for exe in ${BUILD_DIR}/sequential_* ${BUILD_DIR}/parallel_cpu_*; do
    if [[ -x "$exe" ]]; then
        exe_name=$(basename "$exe")
        echo "=== Running $exe_name ==="
        for size in "${SIZES[@]}"; do
            echo "  -> Size: $size"
            output=$("$exe" "$size")
            # Extract time in ms (assuming executable prints "Execution time (ms): X")
            time_ms=$(echo "$output" | grep "Execution time" | awk '{print $4}')
            echo "${exe_name},${size},${time_ms}" >> "${OUTFILE}"
        done
    fi
done

echo "✅ Benchmark finished. Results saved in ${OUTFILE}"
