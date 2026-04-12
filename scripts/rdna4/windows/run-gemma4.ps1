#!/usr/bin/env pwsh
# Launch llama-server with Gemma 4 26B-A4B-it on RDNA 4 Vulkan.
#
# Default config: UD-IQ4_XS (13.4 GB, ~4.25 bpw) at 128K ctx, full GPU, no
# CPU MoE offload. IQ4_XS fits natively in 16 GB VRAM alongside the q8_0
# KV cache and compute buffer, so we skip -ncmoe and take the ~10x prefill
# and ~4x decode win over the ncmoe path. See bench-results table in the
# README for the speed comparison across configs.
#
# Overrides via env var (all optional):
#   TQ_MODEL_PATH  path to a different GGUF
#   TQ_CTX         context size (default 131072). Up to ~131K fits full
#                  GPU with IQ4_XS; 256K requires dropping to IQ3_S or
#                  switching to ncmoe + a larger quant.
#   TQ_CTK / TQ_CTV  KV cache quant (default q8_0). Turbo3 is broken on
#                  Vulkan — see README "Known issues".
#   TQ_BATCH / TQ_UBATCH  llama.cpp -b / -ub (default 128 / 64). Smaller
#                  is safe; larger can blow the compute buffer.
#   TQ_THREADS     CPU threads (default 16, matches 9800X3D 8P/16T).
#   TQ_PORT / TQ_HOST  server bind.
#
# Key flags:
#   -ctk q8_0 -ctv q8_0   Symmetric required by Vulkan FA path. Turbo3 is
#                         broken on Vulkan; q8_0 is the correct default.
#
#   -fa on                Flash attention. Scalar path supports q8_0 and
#                         handles Gemma 4's variable head dims correctly.
#
#   -ngl 99               All layers on GPU. IQ4_XS fits natively.
#
#   -b 128 -ub 64         Safe default. Default (-b 2048 -ub 512) balloons
#                         the compute buffer on older llama.cpp versions.
#
#   -fit off              Disable the --fit auto-tuner. It crashes during
#                         its probe pass on Gemma 4.
#
#   --no-warmup           Skip the empty-batch warmup pass.
#
#   --threads 16          Zen 5 9800X3D = 8P/16T.
#
# VRAM budget target (16 GB RX 9070 XT, full GPU, IQ4_XS @ 128K):
#   IQ4_XS model buffer (Vulkan0)   ~13.4 GB
#   KV cache global q8_0 @ 128K     ~1.36 GB
#   KV cache SWA q8_0               ~0.13 GB
#   Compute buffer (b=128 ub=64)    ~0.34 GB
#   Vulkan driver overhead          ~0.50 GB
#   TOTAL                           ~15.7 GB
#
#   With 15.4 GB usable VRAM this is TIGHT. If you hit OOM, fall back to
#   TQ_CTX=98304 (96K) or switch to IQ3_S for more margin.
#
# System RAM target (32 GB): minimal — model is GPU-resident, no expert
# REPACK, so the llama-server working set stays under ~2 GB. OS + Claude
# Code + browser run unconstrained.

$ErrorActionPreference = "Stop"

# --- Paths ---
$repoRoot     = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$llamaServer  = Join-Path $repoRoot "build-vulkan\bin\Release\llama-server.exe"
$defaultModel = "E:\models\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"

# --- Config (override via env vars) ---
$model   = if ($env:TQ_MODEL_PATH) { $env:TQ_MODEL_PATH } else { $defaultModel }
$port    = if ($env:TQ_PORT)       { $env:TQ_PORT }       else { "8001" }
$host_ip = if ($env:TQ_HOST)       { $env:TQ_HOST }       else { "127.0.0.1" }
$ctx     = if ($env:TQ_CTX)        { $env:TQ_CTX }        else { "131072" }
$batch   = if ($env:TQ_BATCH)      { $env:TQ_BATCH }      else { "128" }
$ubatch  = if ($env:TQ_UBATCH)     { $env:TQ_UBATCH }     else { "64" }
$ctk      = if ($env:TQ_CTK)       { $env:TQ_CTK }       else { "q8_0" }
$ctv      = if ($env:TQ_CTV)       { $env:TQ_CTV }       else { "q8_0" }
$threads  = if ($env:TQ_THREADS)   { $env:TQ_THREADS }   else { "16" }
$slotDir  = if ($env:TQ_SLOT_DIR)  { $env:TQ_SLOT_DIR }  else { "E:\work\slots" }
$slotFile = if ($env:TQ_SLOT_FILE) { $env:TQ_SLOT_FILE } else { "slot-0.bin" }

# --- Preflight ---
if (-not (Test-Path $llamaServer)) {
    Write-Host "llama-server.exe not found at: $llamaServer" -ForegroundColor Red
    Write-Host "Run .\scripts\rdna4\windows\build.ps1 first." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $model)) {
    Write-Host "Model not found at: $model" -ForegroundColor Red
    Write-Host "Download the GGUF and set TQ_MODEL_PATH, or place at default path." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $slotDir)) {
    New-Item -Path $slotDir -ItemType Directory -Force | Out-Null
}

# --- Summary ---
Write-Host "=== Gemma 4 26B-A4B (Vulkan, full GPU) ===" -ForegroundColor Cyan
Write-Host "  model        : $model"
Write-Host "  context      : $ctx tokens"
Write-Host "  KV cache     : K=$ctk V=$ctv (symmetric required for Vulkan FA)"
Write-Host "  batch/ubatch : $batch / $ubatch"
Write-Host "  slot cache   : $slotDir\$slotFile"
Write-Host "  listen       : http://$host_ip`:$port"
Write-Host ""

# --- Launch llama-server in background ---
# ncmoe disabled for IQ4_XS full-GPU path at 128K ctx. To run UD-Q4_K_XL
# at 256K, set TQ_MODEL_PATH + TQ_CTX=262144 and pass -ncmoe 30 via $args.
#
# --slot-save-path enables the /slots/{id}?action=save|restore HTTP API.
# --cache-reuse 256 lets the in-memory prompt cache reuse chunks of 256+
# tokens across requests via KV shifting (good for agentic coding where
# the prefix rarely changes).
$serverArgs = @(
    "-m", $model,
    "--alias", "gemma-4-26b",
    "-ctk", $ctk, "-ctv", $ctv,
    "-fa", "on",
    "-ngl", "99",
    "-c", $ctx,
    "-b", $batch, "-ub", $ubatch,
    "--no-warmup",
    "-fit", "off",
    "--threads", $threads,
    "-np", "1",
    "--slot-save-path", $slotDir,
    "--cache-reuse", "256",
    "--host", $host_ip, "--port", $port
) + $args

$server = Start-Process -FilePath $llamaServer -ArgumentList $serverArgs -PassThru -NoNewWindow

# --- Cleanup handler: save slot state before exit ---
$save = {
    try {
        Write-Host ""
        Write-Host "Saving slot 0 to $slotFile ..." -ForegroundColor Cyan
        $resp = Invoke-RestMethod -Method Post `
            -Uri "http://127.0.0.1:$port/slots/0?action=save" `
            -ContentType "application/json" `
            -Body "{`"filename`":`"$slotFile`"}" `
            -TimeoutSec 30 -ErrorAction Stop
        $tokens = $resp.n_saved
        $mib    = [math]::Round($resp.n_written / 1MB, 0)
        Write-Host "  saved $tokens tokens ($mib MiB)" -ForegroundColor Green
    } catch {
        Write-Host "  save failed: $_" -ForegroundColor Yellow
    }
    if ($server -and -not $server.HasExited) {
        $server.Kill()
    }
}

# Register handler for clean exits (Ctrl+C in PowerShell triggers the finally block)
try {
    # --- Wait for server to come up, then restore if cache present ---
    $healthUrl = "http://127.0.0.1:$port/health"
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        if ($server.HasExited) {
            Write-Host "llama-server exited during startup (code $($server.ExitCode))" -ForegroundColor Red
            exit 1
        }
        try {
            $h = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 -ErrorAction Stop
            if ($h.status -eq "ok") {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $ready) {
        Write-Host "Server did not become ready within 60 seconds" -ForegroundColor Red
        & $save
        exit 1
    }

    Write-Host "Server ready. http://${host_ip}:${port}" -ForegroundColor Green

    # Auto-restore slot cache if file exists
    $slotPath = Join-Path $slotDir $slotFile
    if (Test-Path $slotPath) {
        try {
            Write-Host "Restoring slot 0 from $slotFile ..." -ForegroundColor Cyan
            $resp = Invoke-RestMethod -Method Post `
                -Uri "http://127.0.0.1:$port/slots/0?action=restore" `
                -ContentType "application/json" `
                -Body "{`"filename`":`"$slotFile`"}" `
                -TimeoutSec 60 -ErrorAction Stop
            $tokens = $resp.n_restored
            Write-Host "  restored $tokens tokens from prior session" -ForegroundColor Green
        } catch {
            Write-Host "  restore failed: $_" -ForegroundColor Yellow
        }
    } else {
        Write-Host "No prior slot cache at $slotPath (first run)" -ForegroundColor DarkGray
    }

    Write-Host ""
    Write-Host "Ready for requests. Press Ctrl+C to stop (slot will be auto-saved)." -ForegroundColor Cyan
    Write-Host ""

    # Wait on server
    Wait-Process -Id $server.Id
} finally {
    & $save
}
