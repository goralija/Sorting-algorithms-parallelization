#!/bin/bash
# run_executables.sh
# Rebuilds all executables, runs only changed ones, and logs results
# Adds benchmarking for multiple array types

# Parse command-line arguments for algorithm filters
ALGORITHM_FILTERS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --*)
            ALGORITHM_FILTERS+=("${1#--}")
            shift
            ;;
        *)
            shift
            ;;
    esac
done

CONFIG_FILE="config.yaml"
EXAMPLE_FILE="config.yaml.example"

# Load configuration from YAML
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "⚠️  ${CONFIG_FILE} not found, using ${EXAMPLE_FILE} instead."
    CONFIG_FILE="${EXAMPLE_FILE}"
fi

if ! command -v yq &> /dev/null; then
    echo "⚠️  'yq' is not installed. Install it via:"
    echo "   brew install yq  (macOS) or sudo apt install yq (Linux)"
    exit 1
fi

# Read sizes, types, seed, print flag, benchmark mode from YAML
SIZES=($(yq -r '.sizes[]' "${CONFIG_FILE}"))
TYPES=($(yq -r '.types[]' "${CONFIG_FILE}"))
SEED=$(yq -r '.seed // 12345' "${CONFIG_FILE}")
PRINT_ARRAY=$(yq -r '.print_array // false' "${CONFIG_FILE}")

echo "📘 Loaded configuration:"
echo "  SIZES: ${SIZES[@]}"
echo "  TYPES: ${TYPES[@]}"
echo "  SEED: ${SEED}"
echo "  PRINT_ARRAY: ${PRINT_ARRAY}"

if [[ ${#ALGORITHM_FILTERS[@]} -gt 0 ]]; then
    echo "  Algorithm filters: ${ALGORITHM_FILTERS[@]}"
fi

# Function to check if an executable matches the algorithm filters
test_algorithm_filter() {
    local exe_name=$1
    local exe_lower=$(echo "$exe_name" | tr '[:upper:]' '[:lower:]')
    
    # If no filters specified, run all
    if [[ ${#ALGORITHM_FILTERS[@]} -eq 0 ]]; then
        return 0
    fi
    
    for filter in "${ALGORITHM_FILTERS[@]}"; do
        local filter_lower=$(echo "$filter" | tr '[:upper:]' '[:lower:]')
        
        # Check algorithm type filters
        if [[ "$filter_lower" == "quick" ]] && [[ "$exe_lower" == *"quick"* ]]; then return 0; fi
        if [[ "$filter_lower" == "merge" ]] && [[ "$exe_lower" == *"merge"* ]]; then return 0; fi
        if [[ "$filter_lower" == "bitonic" ]] && [[ "$exe_lower" == *"bitonic"* ]]; then return 0; fi
        if [[ "$filter_lower" == "std" ]] && [[ "$exe_lower" == *"std"* ]]; then return 0; fi
        if [[ "$filter_lower" == "radix" ]] && [[ "$exe_lower" == *"radix"* ]]; then return 0; fi
        
        # Check implementation category filters
        if [[ "$filter_lower" == "seq-naive" ]] && [[ "$exe_lower" == sequential_naive* ]]; then return 0; fi
        if [[ "$filter_lower" == "seq-optimized" ]] && [[ "$exe_lower" == sequential_optimized* ]]; then return 0; fi
        if [[ "$filter_lower" == "cpu-parallel" ]] && [[ "$exe_lower" == parallel_cpu* ]]; then return 0; fi
        
        # Check exact name match
        if [[ "$exe_lower" == *"$filter_lower"* ]]; then return 0; fi
    done
    
    return 1
}

DATA_DIR="data"
HASH_FILE="${DATA_DIR}/last_run_hashes.txt"
OUTFILE="${DATA_DIR}/benchmark.csv"

# Always build both versions
BUILD_DIR_NORMAL="build"
BUILD_DIR_BENCHMARK="build_vtune_benchmarking"

echo ""
echo "📁 Building both versions:"
echo "   • Normal (with verification): ${BUILD_DIR_NORMAL}"
echo "   • Benchmark (VTune ready): ${BUILD_DIR_BENCHMARK}"

CORES=$(sysctl -n hw.logicalcpu)

# Function to build executables
build_executables() {
    local build_dir=$1
    local benchmark_flag=$2
    local description=$3
    
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "🔧 Building ${description}"
    echo "═══════════════════════════════════════════════════════════"
    
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"
    
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
          -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
          ${benchmark_flag} ..
    
    if [[ $? -ne 0 ]]; then
        echo "❌ CMake configuration failed for ${description}"
        cd ..
        return 1
    fi
    
    make -j${CORES}
    
    if [[ $? -ne 0 ]]; then
        echo "❌ Build failed for ${description}"
        cd ..
        return 1
    fi
    
    cd ..
    echo "✅ Build completed successfully!"
    return 0
}

# Build both versions
build_executables "${BUILD_DIR_NORMAL}" "-DBENCHMARK_MODE=OFF" "NORMAL executables (with verification)"
NORMAL_SUCCESS=$?

build_executables "${BUILD_DIR_BENCHMARK}" "-DBENCHMARK_MODE=ON" "BENCHMARK executables (VTune ready)"
BENCHMARK_SUCCESS=$?

if [[ ${NORMAL_SUCCESS} -ne 0 ]] || [[ ${BENCHMARK_SUCCESS} -ne 0 ]]; then
    echo ""
    echo "❌ One or more builds failed."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Both versions built successfully!"
echo "   📂 Normal: ${BUILD_DIR_NORMAL} (will be executed)"
echo "   📂 Benchmark: ${BUILD_DIR_BENCHMARK} (ready for VTune)"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "${DATA_DIR}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "🚀 Running normal executables from: ${BUILD_DIR_NORMAL}"
echo "═══════════════════════════════════════════════════════════"

# Backup current benchmark if it exists
if [[ -f "${OUTFILE}" ]]; then
    mv "${OUTFILE}" "${DATA_DIR}/benchmark_backup_$(date +%s).csv"
fi

# Find latest benchmark backup (if exists)
LATEST_BACKUP=$(ls -t ${DATA_DIR}/benchmark_backup_*.csv 2>/dev/null | head -n 1)

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

# Compute new hashes and decide which executables changed (only from normal build)
for exe in ${BUILD_DIR_NORMAL}/sequential_* ${BUILD_DIR_NORMAL}/parallel_cpu_*; do
    if [[ -x "$exe" ]]; then
        exe_name=$(basename "$exe")
        hash=$(md5 -q "$exe" 2>/dev/null || md5sum "$exe" | awk '{print $1}')
        echo "$exe $hash" >> "${TMP_HASH_FILE}"

        old_hash=$(get_old_hash "$exe")
        
        # Check if this executable matches the algorithm filters
        if test_algorithm_filter "$exe_name"; then
            matches_filter=true
        else
            matches_filter=false
        fi

        if [[ "$hash" == "$old_hash" ]]; then
            if [[ "$matches_filter" == "true" ]]; then
                echo "⏭️  Skipping unchanged (matches filter): $exe_name"
            else
                echo "⏭️  Skipping (no filter match): $exe_name"
            fi
            
            # Ako postoji stari benchmark, kopiraj rezultate
            echo "  🔄 Copying previous results for: $exe_name"
            if [[ -n "${LATEST_BACKUP}" ]]; then
                grep "^${exe_name}," "${LATEST_BACKUP}" >> "${OUTFILE}"
            fi
            continue
        fi
        
        if [[ "$matches_filter" == "false" ]]; then
            echo "⏭️  Skipping (no filter match): $exe_name"
            # Copy previous results even if executable changed but doesn't match filter
            if [[ -n "${LATEST_BACKUP}" ]]; then
                grep "^${exe_name}," "${LATEST_BACKUP}" >> "${OUTFILE}"
            fi
            continue
        fi

        echo "🚀 Running updated executable: $exe_name"
        for type in "${TYPES[@]}"; do
            for size in "${SIZES[@]}"; do
                echo "  -> Type: $type | Size: $size"
                # build args (size, type, seed, optional flags)
                args=("$size" "$type" "$SEED")
                if [[ "${PRINT_ARRAY}" == "true" ]]; then
                    args+=("--print-array")
                fi
                # Note: benchmark mode is now compile-time, no runtime flag needed

                output=$("$exe" "${args[@]}" 2>&1)

                # Print full output and parse results
                # Print full output to terminal for debugging
                echo "----- ${exe_name} output start -----"
                echo "$output"
                echo "----- ${exe_name} output end -----"

                # Check if output contains sorting error
                if echo "$output" | grep -q "Error: Array is NOT sorted"; then
                    echo "❗ Skipping invalid result for $exe_name (unsorted output)"
                    continue
                fi
                
                time_ms=$(echo "$output" | grep "Execution time" | awk '{print $4}')
                
                # Verify we got a valid time measurement
                if [[ -z "$time_ms" ]]; then
                    echo "❗ No valid execution time found for $exe_name"
                    continue
                fi

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

