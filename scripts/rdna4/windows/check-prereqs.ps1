#!/usr/bin/env pwsh
# Preflight check for RDNA 4 Vulkan build + Gemma 4 run.
# Verifies toolchain, driver, VRAM, model file, system RAM.
# Run from "x64 Native Tools Command Prompt for VS 2022" to get MSVC env.

$ErrorActionPreference = "Stop"

$ok = $true
function Check($label, $test, $fix) {
    Write-Host -NoNewline "  $label ... "
    try {
        $result = & $test
        if ($result) {
            Write-Host "ok" -ForegroundColor Green
            return $true
        } else {
            Write-Host "MISSING" -ForegroundColor Red
            Write-Host "    fix: $fix" -ForegroundColor Yellow
            $script:ok = $false
            return $false
        }
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
        Write-Host "    fix: $fix" -ForegroundColor Yellow
        $script:ok = $false
        return $false
    }
}

Write-Host "=== RDNA 4 / Vulkan / Gemma 4 preflight ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "Toolchain:"
Check "MSVC cl.exe" { (Get-Command cl -ErrorAction SilentlyContinue) -ne $null } "Open 'x64 Native Tools Command Prompt for VS 2022', or install VS 2022 Build Tools with C++ workload"
Check "cmake >= 3.25" {
    $v = (cmake --version 2>&1 | Select-Object -First 1) -replace 'cmake version ', ''
    [version]$v -ge [version]"3.25.0"
} "winget install Kitware.CMake (then restart shell)"
Check "git" { (Get-Command git -ErrorAction SilentlyContinue) -ne $null } "winget install Git.Git"
Check "VULKAN_SDK env" { $env:VULKAN_SDK -ne $null -and (Test-Path $env:VULKAN_SDK) } "Install from https://vulkan.lunarg.com (Windows installer sets VULKAN_SDK automatically)"
Check "glslc shader compiler" { (Get-Command glslc -ErrorAction SilentlyContinue) -ne $null } "Reinstall Vulkan SDK and ensure %VULKAN_SDK%\Bin is in PATH"

Write-Host ""
Write-Host "GPU / Driver:"
Check "vulkaninfo" { (Get-Command vulkaninfo -ErrorAction SilentlyContinue) -ne $null } "Install AMD Adrenalin 25.x driver from amd.com/support"
Check "RX 9700 XT detected" {
    $info = vulkaninfo --summary 2>$null | Out-String
    $info -match "Radeon RX 9700 XT" -or $info -match "gfx1201"
} "Check AMD Adrenalin driver is installed and GPU is seated"

Write-Host ""
Write-Host "System:"
$totalRam = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Check "RAM >= 32 GB (have: $totalRam GB)" { $totalRam -ge 31 } "Gemma 4 26B expert offload needs ~22 GB working set"

Write-Host ""
Write-Host "Model file:"
$defaultModel = if ($env:TQ_MODEL_PATH) { $env:TQ_MODEL_PATH } else { "C:\models\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" }
Check "GGUF present ($defaultModel)" { Test-Path $defaultModel } "Download from https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF (UD-Q4_K_XL, ~17.1 GB) and place at the path above, or set TQ_MODEL_PATH"

Write-Host ""
Write-Host "Build output:"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$llamaServer = Join-Path $repoRoot "build-vulkan\bin\Release\llama-server.exe"
Check "llama-server.exe built" { Test-Path $llamaServer } "Run .\scripts\rdna4\windows\build.ps1 first"

Write-Host ""
if ($ok) {
    Write-Host "All checks passed." -ForegroundColor Green
    exit 0
} else {
    Write-Host "One or more checks failed. See messages above." -ForegroundColor Red
    exit 1
}
