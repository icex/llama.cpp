#!/usr/bin/env pwsh
# Build llama-cpp-turboquant with Vulkan backend for RDNA 4 (RX 9700 XT / gfx1201).
# Run from "x64 Native Tools Command Prompt for VS 2022".

$ErrorActionPreference = "Stop"

$repoRoot  = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$buildDir  = Join-Path $repoRoot "build-vulkan"

if (-not $env:VULKAN_SDK -or -not (Test-Path $env:VULKAN_SDK)) {
    Write-Host "VULKAN_SDK not set or path missing." -ForegroundColor Red
    Write-Host "Install from https://vulkan.lunarg.com and restart the shell." -ForegroundColor Yellow
    exit 1
}

Write-Host "Configuring cmake (Vulkan, Release)..." -ForegroundColor Cyan
cmake -B $buildDir -S $repoRoot `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_VULKAN=ON `
    -DGGML_NATIVE=ON `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_TOOLS=ON
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building..." -ForegroundColor Cyan
cmake --build $buildDir --config Release --target llama-server llama-cli llama-bench --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$llamaServer = Join-Path $buildDir "bin\Release\llama-server.exe"
if (Test-Path $llamaServer) {
    Write-Host ""
    Write-Host "Built: $llamaServer" -ForegroundColor Green
    & $llamaServer --version 2>&1 | Select-Object -First 3
} else {
    Write-Host "Build succeeded but llama-server.exe not found at expected path." -ForegroundColor Red
    exit 1
}
