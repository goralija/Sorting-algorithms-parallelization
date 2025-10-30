# run_executables.ps1
# Rebuilds all executables, runs only changed ones, and logs results

$BuildDir = "build"
$DataDir = "data"
$HashFile = "$DataDir\last_run_hashes.txt"
$OutFile = "$DataDir\benchmark.csv"
$Sizes = @(5000000, 50000000, 500000000)

# Detect number of CPU cores
$Cores = [Environment]::ProcessorCount

Write-Host "🔧 Rebuilding executables..."
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
New-Item -ItemType Directory -Path $BuildDir | Out-Null
Push-Location $BuildDir
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -- /m:$Cores
Pop-Location

if (-Not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir | Out-Null }

# Backup previous benchmark file
if (Test-Path $OutFile) {
    $timestamp = Get-Date -UFormat %s
    Move-Item $OutFile "$DataDir\benchmark_backup_$timestamp.csv"
}
"Algorithm,ArraySize,TimeMs" | Out-File $OutFile

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

# Compute new hashes
$newHashes = @{}
Get-ChildItem $BuildDir -File | Where-Object { $_.Name -match '^(sequential_|parallel_cpu_)' } | ForEach-Object {
    $exePath = $_.FullName
    $hash = (Get-FileHash $exePath -Algorithm MD5).Hash
    $newHashes[$exePath] = $hash
}

# Run only changed executables
foreach ($exePath in $newHashes.Keys) {
    $exeName = Split-Path $exePath -Leaf
    $oldHash = if ($oldHashes.ContainsKey($exePath)) { $oldHashes[$exePath] } else { "" }
    $newHash = $newHashes[$exePath]

    if ($oldHash -eq $newHash) {
        Write-Host "⏭️  Skipping unchanged: $exeName"
        continue
    }

    Write-Host "🚀 Running updated executable: $exeName"
    foreach ($size in $Sizes) {
        Write-Host "  -> Size: $size"
        $output = & $exePath $size
        $timeLine = ($output | Select-String -Pattern "Execution time").Line
        if ($timeLine) {
            $timeMs = ($timeLine -split " ")[-1]
            "$exeName,$size,$timeMs" | Out-File $OutFile -Append
        } else {
            Write-Host "⚠️  Warning: Could not parse time output for $exeName ($size)"
        }
    }
}

# Save new hashes
$newHashes.GetEnumerator() | ForEach-Object {
    "$($_.Key) $($_.Value)" 
} | Out-File $HashFile

Write-Host "✅ Benchmark finished. Results saved in $OutFile"
