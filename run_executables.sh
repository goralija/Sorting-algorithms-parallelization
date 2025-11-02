#!/bin/bash
# run_executables.sh
# Rebuilds all executables, runs only changed ones, and logs results
# Copies results of unchanged executables from previous benchmark
# Compatible with macOS default bash (v3.2)

BUILD_DIR="build"
DATA_DIR="data"
HASH_FILE="${DATA_DIR}/last_run_hashes.txt"
OUTFILE="${DATA_DIR}/benchmark.csv"
SIZES=(10000 5000000 50000000 500000000)

# Detect number of CPU cores (macOS compatible)
CORES=$(sysctl -n hw.logicalcpu)

echo "🔧 Rebuilding executables..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ ..
make -j${CORES}
cd ..

mkdir -p "${DATA_DIR}"

# Find latest benchmark backup (if exists)
LATEST_BACKUP=$(ls -t ${DATA_DIR}/benchmark_backup_*.csv 2>/dev/null | head -n 1)

# Backup current benchmark if it exists
if [[ -f "${OUTFILE}" ]]; then
    mv "${OUTFILE}" "${DATA_DIR}/benchmark_backup_$(date +%s).csv"
fi

# Initialize new CSV
echo "Algorithm,ArraySize,TimeMs" > "${OUTFILE}"

# Load old hashes if exist
touch "${HASH_FILE}"
TMP_HASH_FILE="${DATA_DIR}/new_hashes.txt"
> "${TMP_HASH_FILE}"

# Function to get old hash for a given exe
get_old_hash() {
    grep "^$1 " "${HASH_FILE}" | awk '{print $2}'
}

# Compute new hashes and decide which executables changed
for exe in ${BUILD_DIR}/sequential_* ${BUILD_DIR}/parallel_cpu_*; do
    if [[ -x "$exe" ]]; then
        exe_name=$(basename "$exe")
        hash=$(md5 -q "$exe" 2>/dev/null || md5sum "$exe" | awk '{print $1}')
        echo "$exe $hash" >> "${TMP_HASH_FILE}"

        old_hash=$(get_old_hash "$exe")

        if [[ "$hash" == "$old_hash" ]]; then
            echo "⏭️  Skipping unchanged: $exe_name"

            # Ako postoji stari benchmark, kopiraj rezultate
            if [[ -n "${LATEST_BACKUP}" ]]; then
                grep "^${exe_name}," "${LATEST_BACKUP}" >> "${OUTFILE}"
            fi
            continue
        fi

        echo "🚀 Running updated executable: $exe_name"
        for size in "${SIZES[@]}"; do
            echo "  -> Size: $size"
            output=$("$exe" "$size" 2>&1)
            time_ms=$(echo "$output" | grep "Execution time" | awk '{print $4}')
            echo "${exe_name},${size},${time_ms}" >> "${OUTFILE}"
        done
    fi
done

# Save new hashes
mv "${TMP_HASH_FILE}" "${HASH_FILE}"

echo "✅ Benchmark finished. Results saved in ${OUTFILE}"