# run_executables.ps1
# Rebuilds all executables, runs only changed ones, and logs results
# Copies results of unchanged executables from previous benchmark

$ConfigFile = "config.yaml"
$ExampleFile = "config.yaml.example"

if (-not (Test-Path $ConfigFile)) {
    Write-Host "⚠️  $ConfigFile not found, using $ExampleFile instead."
    $ConfigFile = $ExampleFile
}

# Check if YAML support exists (PowerShell 7+ has ConvertFrom-Yaml)
if (-not (Get-Command ConvertFrom-Yaml -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️  ConvertFrom-Yaml not available. Install module:"
    Write-Host "   Install-Module powershell-yaml -Scope CurrentUser"
    exit 1
}

# Load config
$config = Get-Content $ConfigFile | ConvertFrom-Yaml

$Sizes = $config.sizes
$Types = $config.types
$Seed = if ($null -ne $config.seed) { [int]$config.seed } else { 12345 }
$PrintArray = if ($null -ne $config.print_array) { [bool]$config.print_array } else { $false }
$BenchmarkMode = if ($null -ne $config.benchmark_measurement) { [bool]$config.benchmark_measurement } else { $false }

Write-Host "📘 Loaded configuration:"
Write-Host "  Sizes: $($Sizes -join ', ')"
Write-Host "  Types: $($Types -join ', ')"
Write-Host "  Seed: $Seed"
Write-Host "  Print array: $PrintArray"
Write-Host "  Benchmark mode: $BenchmarkMode"

# Use separate build directories for normal vs benchmark builds
$BuildDir = if ($BenchmarkMode) { "build_vtune_benchmarking" } else { "build" }
$DataDir = "data"
$HashFile = "$DataDir\last_run_hashes.txt"
$OutFile = "$DataDir\benchmark.csv"

Write-Host "📁 Build directory: $BuildDir"

# Detect number of CPU cores
$Cores = [Environment]::ProcessorCount

Write-Host "🔧 Rebuilding executables..."
if (Test-Path $BuildDir) { 
    # pokušaj nekoliko puta zbog "used by another process" grešaka
    Write-Host "🗑️  Removing existing build directory..."
    for ($i=0; $i -lt 5; $i++) {
        try {
            Remove-Item -Recurse -Force $BuildDir
            break
        } catch {
            Start-Sleep -Milliseconds 200
        }
    }
}
if (-Not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }

Push-Location $BuildDir

Write-Host "🧱 Configuring CMake build..."

# Build CMake benchmark flag based on config
$BenchmarkFlag = if ($BenchmarkMode) { "-DBENCHMARK_MODE=ON" } else { "-DBENCHMARK_MODE=OFF" }

# 1️⃣ Preferred configuration (GCC + Ninja)
Write-Host "🔧 Trying preferred configuration (Ninja + GCC)..."
$preferredCommand = "cmake -G `"Ninja`" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe -DUSE_OPENMP=ON $BenchmarkFlag -DCMAKE_CXX_FLAGS=`"-O3 -march=native -mavx2 -mavx512f -fopenmp`" .."
Invoke-Expression $preferredCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Preferred configuration succeeded!"
} else {
    Write-Host "❌ Preferred configuration failed, trying alternative generators..."
    Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue

    # 2️⃣ Fallback options in order of preference
    $generators = @(
        @("MinGW Makefiles", "cmake .. -G `"MinGW Makefiles`" -DCMAKE_BUILD_TYPE=Release $BenchmarkFlag"),
        @("Ninja with Clang", "cmake .. -G `"Ninja`" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release $BenchmarkFlag"),
        @("Visual Studio Clang", "cmake .. -G `"Visual Studio 17 2022`" -A x64 -T ClangCL $BenchmarkFlag"),
        @("Default", "cmake .. -DCMAKE_BUILD_TYPE=Release $BenchmarkFlag")
    )

    $success = $false
    foreach ($generator in $generators) {
        $name = $generator[0]
        $command = $generator[1]
        Write-Host "Trying $name..."
        Invoke-Expression $command

        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Configuration successful with $name"
            $success = $true
            break
        } else {
            Write-Host "❌ $name failed, trying next..."
            Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue
        }
    }

    if (-not $success) {
        Write-Host "❌ All CMake configuration attempts failed."
        Pop-Location
        exit 1
    }
}

# 3️⃣ Build using all CPU cores
Write-Host "⚙️ Building project with all available cores ($Cores)..."
cmake --build . --config Release --parallel $Cores

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed."
    Pop-Location
    exit 1
}

# 4️⃣ Copy required DLLs to build directory for VTune and standalone execution
Write-Host "📦 Copying runtime DLLs to build directory..."
$dllsToCopy = @(
    "C:\msys64\mingw64\bin\libstdc++-6.dll",
    "C:\msys64\mingw64\bin\libgcc_s_seh-1.dll",
    "C:\msys64\mingw64\bin\libwinpthread-1.dll",
    "C:\msys64\mingw64\bin\libgomp-1.dll"  # OpenMP runtime
)
foreach ($dll in $dllsToCopy) {
    if (Test-Path $dll) {
        Copy-Item $dll -Destination . -Force
        Write-Host "  ✅ Copied $(Split-Path $dll -Leaf)"
    }
}

Pop-Location
Write-Host "✅ Build completed successfully!"

# In benchmark mode, don't execute - VTune will run them
if ($BenchmarkMode) {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════"
    Write-Host "🎯 Benchmark executables built in: $BuildDir"
    Write-Host "📊 Ready for VTune profiling - executables NOT executed"
    Write-Host "💡 Run them manually through VTune or directly for profiling"
    Write-Host "═══════════════════════════════════════════════════════════"
    exit 0
}

if (-Not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }

# Backup current benchmark if exists
if (Test-Path $OutFile) {
    $timestamp = Get-Date -UFormat %s
    Move-Item $OutFile "$DataDir\benchmark_backup_$timestamp.csv"
}

# Find latest benchmark backup
$LatestBackup = Get-ChildItem -Path $DataDir -Filter "benchmark_backup_*.csv" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Initialize new CSV
"Algorithm,ArraySize,ArrayType,TimeMs" | Out-File $OutFile

# Load old hashes
$oldHashes = @{}
if (Test-Path $HashFile) {
    Get-Content $HashFile | ForEach-Object {
        $parts = $_ -split " "
        if ($parts.Length -eq 2) {
            $oldHashes[$parts[0]] = $parts[1]
        }
    }
}

# Temporary hash storage
$newHashes = @{}

# Compute new hashes
Get-ChildItem $BuildDir -File | Where-Object { $_.Name -match '^(sequential_|parallel_cpu_)' } | ForEach-Object {
    $exePath = $_.FullName
    $exeName = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $hash = (Get-FileHash $exePath -Algorithm MD5).Hash
    $newHashes[$exePath] = $hash

    $oldHash = if ($oldHashes.ContainsKey($exePath)) { $oldHashes[$exePath] } else { "" }

    if ($oldHash -eq $hash) {
        Write-Host "⏭️  Skipping unchanged: $exeName"
        if ($LatestBackup) {
            $lines = Get-Content $LatestBackup.FullName | Where-Object { $_ -match "^$exeName," }
            if ($lines) {
                $lines | Out-File $OutFile -Append
            }
        }
        return
    }

    Write-Host "🚀 Running updated executable: $exeName"
    foreach ($type in $Types) {
        foreach ($size in $Sizes) {
            Write-Host "  -> Type: $type | Size: $size"
            # build arguments: size, type, seed, optional flags
            $args = @($size, $type, $Seed)
            if ($PrintArray) { $args += "--print-array" }
            # Note: benchmark mode is now compile-time, no runtime flag needed

            # Run executable and capture output
            $output = & $exePath @args 2>&1

            # In benchmark mode, skip detailed output parsing (minimal overhead)
            if ($BenchmarkMode) {
                Write-Host "  ✅ Benchmark mode run completed (no output verification)"
                continue
            }

            # Normal mode: print full output and parse results
            # Print full output to console so array is visible for debugging
            Write-Host "----- $exeName output start -----"
            Write-Host $output
            Write-Host "----- $exeName output end -----"

            if ($output -match "Error: Array is NOT sorted") {
                Write-Host "❗ Skipping invalid result for $exeName (unsorted output)"
                return
            }
            # Parse time (still works even if array printed)
            $timeLine = ($output | Select-String -Pattern "Execution time").Line

            if ($timeLine) {
                # take last token as ms (your existing format)
                $tokens = $timeLine -split '\s+'
                $timeMs = $tokens[-1]
                "$exeName,$size,$type,$timeMs" | Out-File $OutFile -Append
            } else {
                Write-Host "⚠️  Warning: Could not parse time output for $exeName ($size, $type)"
            }
        }
    }
}

# Save new hashes
$newHashes.GetEnumerator() | ForEach-Object {
    "$($_.Key) $($_.Value)"
} | Out-File $HashFile

Write-Host "✅ Benchmark finished. Results saved in $OutFile"

# ============================================================================
# Python plotting setup and execution
# ============================================================================

# Detect Windows Python (skip MSYS2 Python)
$winPython = Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -notmatch "msys64" }
if (-not $winPython) {
    Write-Host "⚠️  Could not find Windows Python 3.10+. Skipping plotting."
    $RunPlots = $false
} else {
    $RunPlots = $true
    $PythonExe = $winPython.Source
}

Write-Host ""
Write-Host ("=" * 60)
Write-Host "📊 Setting up Python environment for plotting..."
Write-Host ("=" * 60)

$VenvDir = "venv"
$PlotsScript = "plots/plot_results.py"

# Check if Python is available
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "⚠️  Python not found. Skipping plot generation."
    Write-Host "   Install Python 3.10+ to enable visualization."
    exit 0
}

$pythonVersion = & python --version 2>&1
Write-Host "✅ Found $pythonVersion"

# Create virtual environment if it doesn't exist
if (-not (Test-Path $VenvDir) -and $RunPlots) {
    Write-Host "📦 Creating Python virtual environment..."
    & $PythonExe -m venv $VenvDir
    Write-Host "✅ Virtual environment created"
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..."
# If activation fails, just call Python from venv directly
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# Install dependencies
if ((Test-Path "requirements.txt") -and $RunPlots) {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
} elseif ($RunPlots) {
    & $VenvPython -m pip install matplotlib pandas numpy
}

# Run plotting script
if ($RunPlots -and (Test-Path $PlotsScript)) {
    Write-Host ""
    Write-Host "🎨 Generating plots..."
    & $VenvPython $PlotsScript

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host ("=" * 60)
        Write-Host "✅ SUCCESS! All plots generated."
        Write-Host "📂 Check the plots/ directory for visualization results."
        Write-Host ("=" * 60)
    } else {
        Write-Host "⚠️  Plot generation encountered errors."
    }
}

# Note: Virtual environment will be deactivated when script exits

