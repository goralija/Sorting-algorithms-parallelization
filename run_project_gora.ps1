# run_executables.ps1
# Rebuilds all executables, runs only changed ones, and logs results
# Copies results of unchanged executables from previous benchmark

$BuildDir = "build"
$DataDir = "data"
$HashFile = "$DataDir\last_run_hashes.txt"
$OutFile = "$DataDir\benchmark.csv"
$Sizes = @(10000, 5000000, 50000000, 500000000)
$Types = @("random", "sorted", "reversed", "nearly_sorted", "few_unique")

# Detect number of CPU cores
$Cores = [Environment]::ProcessorCount

Write-Host "🔧 Rebuilding executables..."
if (Test-Path $BuildDir) { 
    Write-Host "🗑️  Removing existing build directory..."
    Remove-Item -Recurse -Force $BuildDir 
}
New-Item -ItemType Directory -Path $BuildDir | Out-Null

Push-Location $BuildDir

# In the configuration section:
Write-Host "🧱 Configuring CMake build..."

# Try different generators in order of preference
$generators = @(
    @("MinGW Makefiles", "cmake .. -G `"MinGW Makefiles`" -DCMAKE_BUILD_TYPE=Release"),
    @("Ninja with Clang", "cmake .. -G `"Ninja`" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release"),
    @("Visual Studio Clang", "cmake .. -G `"Visual Studio 17 2022`" -A x64 -T ClangCL"),
    @("Default", "cmake .. -DCMAKE_BUILD_TYPE=Release")
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
        # Clean up failed attempt
        Remove-Item CMakeCache.txt -ErrorAction SilentlyContinue
    }
}

if (-not $success) {
    Write-Host "❌ All CMake configuration attempts failed."
    Pop-Location
    exit 1
}

Write-Host "⚙️ Building project with all available cores ($Cores)..."
# Then build the project
cmake --build . --config Release --parallel $Cores

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed."
    Pop-Location
    exit 1
}

Pop-Location

Write-Host "✅ Build completed successfully!"

# Rest of your script continues here...
if (-Not (Test-Path $DataDir)) { 
    New-Item -ItemType Directory -Path $DataDir | Out-Null 
}

# Find latest benchmark backup
$LatestBackup = Get-ChildItem -Path $DataDir -Filter "benchmark_backup_*.csv" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1

# Backup current benchmark if exists
if (Test-Path $OutFile) {
    $timestamp = Get-Date -UFormat %s
    Move-Item $OutFile "$DataDir\benchmark_backup_$timestamp.csv"
}

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
    $exeName = $_.Name
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
            $output = & $exePath $size $type 2>&1
            $timeLine = ($output | Select-String -Pattern "Execution time").Line
            if ($timeLine) {
                $timeMs = ($timeLine -split " ")[-1]
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
if (-not (Test-Path $VenvDir)) {
    Write-Host "📦 Creating Python virtual environment..."
    python -m venv $VenvDir
    Write-Host "✅ Virtual environment created"
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..."
& "$VenvDir\Scripts\Activate.ps1"

# Install/update dependencies
if (Test-Path "requirements.txt") {
    Write-Host "📥 Installing Python dependencies..."
    python -m pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    Write-Host "✅ Dependencies installed"
} else {
    Write-Host "⚠️  requirements.txt not found. Installing basic packages..."
    pip install --quiet matplotlib pandas numpy
}

# Run plotting script
if (Test-Path $PlotsScript) {
    Write-Host ""
    Write-Host "🎨 Generating plots..."
    python $PlotsScript
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host ("=" * 60)
        Write-Host "✅ SUCCESS! All plots generated."
        Write-Host "📂 Check the plots/ directory for visualization results."
        Write-Host ("=" * 60)
    } else {
        Write-Host "⚠️  Plot generation encountered errors."
    }
} else {
    Write-Host "⚠️  Plotting script not found: $PlotsScript"
}

# Note: Virtual environment will be deactivated when script exits

