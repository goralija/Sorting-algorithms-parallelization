#!/bin/bash
# run_executables.sh
# Rebuilds all executables, runs only changed ones, and logs results
# Adds benchmarking for multiple array types

BUILD_DIR="build"
DATA_DIR="data"
HASH_FILE="${DATA_DIR}/last_run_hashes.txt"
OUTFILE="${DATA_DIR}/benchmark.csv"
SIZES=(10000 5000000 50000000 500000000)
TYPES=("random" "sorted" "reversed" "nearly_sorted" "few_unique")

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
echo "Algorithm,ArraySize,ArrayType,TimeMs" > "${OUTFILE}"

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
        for type in "${TYPES[@]}"; do
            for size in "${SIZES[@]}"; do
                echo "  -> Type: $type | Size: $size"
                output=$("$exe" "$size" "$type" 2>&1)
                time_ms=$(echo "$output" | grep "Execution time" | awk '{print $4}')
                echo "${exe_name},${size},${type},${time_ms}" >> "${OUTFILE}"
            done
        done
    fi
done

# Save new hashes
mv "${TMP_HASH_FILE}" "${HASH_FILE}"

echo "✅ Benchmark finished. Results saved in ${OUTFILE}"

# ============================================================================
# Python plotting setup and execution
# ============================================================================

echo ""
echo "============================================================"
echo "📊 Setting up Python environment for plotting..."
echo "============================================================"
echo ""

VENV_DIR="venv"
PLOTS_SCRIPT="plots/plot_results.py"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python 3 not found. Skipping plot generation."
    echo "   Install Python 3.10+ to enable visualization."
    exit 0
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Found Python ${PYTHON_VERSION}"

# Create virtual environment if it doesn't exist
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Install/update dependencies
if [[ -f "requirements.txt" ]]; then
    echo "📥 Installing Python dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  requirements.txt not found. Installing basic packages..."
    pip install --quiet matplotlib pandas numpy
fi

# Run plotting script
if [[ -f "${PLOTS_SCRIPT}" ]]; then
    echo ""
    echo "🎨 Generating plots..."
    python3 "${PLOTS_SCRIPT}"
    
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "============================================================"
        echo "✅ SUCCESS! All plots generated."
        echo "📂 Check the plots/ directory for visualization results."
        echo "============================================================"
        echo ""
    else
        echo "⚠️  Plot generation encountered errors."
    fi
else
    echo "⚠️  Plotting script not found: ${PLOTS_SCRIPT}"
fi

# Virtual environment will be deactivated when script exits

