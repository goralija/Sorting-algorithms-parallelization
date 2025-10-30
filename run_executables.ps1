# Still in development - modify as needed

# run_executables.ps1
# Rebuilds project and runs executables with multiple input sizes

$BuildDir = "build"
$DataDir = "data"
$Sizes = @(10000, 100000, 500000, 1000000, 5000000)

# Clean rebuild
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
New-Item -ItemType Directory -Path $BuildDir
Push-Location $BuildDir
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
Pop-Location

# Prepare CSV
if (-Not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir }
$BenchmarkFile = "$DataDir\benchmark.csv"
"Algorithm,Mode,ArraySize,TimeMs" | Out-File $BenchmarkFile

# Run executables
Get-ChildItem $BuildDir | Where-Object { $_.Extension -eq ".exe" } | ForEach-Object {
    $exePath = $_.FullName
    $exeName = $_.Name
    Write-Host "=== Running $exeName ==="
    foreach ($size in $Sizes) {
        Write-Host "  -> Size: $size"
        $output = & $exePath $size
        $timeLine = ($output | Select-String -Pattern "Execution time").Line
        $timeMs = ($timeLine -split " ")[-1]
        "$exeName,auto,$size,$timeMs" | Out-File $BenchmarkFile -Append
    }
}

Write-Host "✅ Benchmark finished. Results saved in $BenchmarkFile"
