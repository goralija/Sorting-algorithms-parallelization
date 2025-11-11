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


# FORCE GCC generator
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe -DUSE_OPENMP=ON -DCMAKE_CXX_FLAGS="-O3 -march=native -mavx2 -mavx512f -fopenmp" ..
cmake --build . --config Release #-- /m:$Cores
Pop-Location