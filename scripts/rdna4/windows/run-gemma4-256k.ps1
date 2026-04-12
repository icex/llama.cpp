#!/usr/bin/env pwsh
# Launch llama-server with Gemma 4 26B-A4B-it at 256K context on RDNA 4 Vulkan.
#
# Config rationale (see scripts/rdna4/windows/README.md for full analysis):
#
#   -ctk turbo3 -ctv turbo3   Symmetric required — Vulkan FA path does not
#                             support mixed K/V quant types. turbo3 is the
#                             only turbo flavor with Vulkan shaders.
#
#   -fa on                    Flash attention. Scalar + cm1 + cm2 FA shaders
#                             all ship turbo3 pipeline variants.
#
#   -ngl 99                   Offload all layers' structural tensors to GPU
#                             (attention, embeddings, norms).
#
#   -ncmoe 30                 Override expert FFN tensors on all 30 MoE
#                             layers back to CPU. Gemma 4 26B-A4B has ~14 GB
#                             of expert weights, which blows the 16 GB VRAM
#                             budget. Experts compute on CPU, attention on
#                             GPU, activations cross via small PCIe copies.
#
#   -c 262144                 Full 256K context.
#
#   -b 128 -ub 64             CRITICAL. Compute buffer is batch-driven, not
#                             ctx-driven. Default (-b 2048 -ub 512) allocates
#                             ~32 GB of compute buffer and instant-OOMs on
#                             16 GB VRAM. b=128 ub=64 keeps it at ~8 GB.
#
#   --mlock                   Pin expert weights in RAM. Prevents Windows
#                             from paging experts to disk under memory
#                             pressure. Without this, per-token decode can
#                             stall for seconds.
#
#   --no-warmup               Skip the empty-batch warmup pass. Warmup can
#                             trigger OOM on tight VRAM budgets.
#
#   --threads 16              Zen 5 9800X3D = 8P/16T. CPU-side expert FFN
#                             is memory-bandwidth bound; more threads past
#                             8 rarely help but don't hurt.
#
# VRAM budget target (16 GB RX 9700 XT):
#   non-expert model   ~2.7 GB
#   KV cache turbo3    ~1.0 GB  (1 GB global + 49 MB SWA)
#   compute buffer     ~8.3 GB
#   driver overhead    ~0.5 GB
#   TOTAL              ~12.5 GB   (~3.5 GB headroom)
#
# System RAM target (32 GB):
#   expert REPACK      ~8.2 GB   (CPU backend reformats experts)
#   mmap hot set       ~9.0 GB
#   OS + apps          ~8.0 GB
#   TOTAL              ~25.2 GB  (close Electron apps during runs)

$ErrorActionPreference = "Stop"

# --- Paths ---
$repoRoot     = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$llamaServer  = Join-Path $repoRoot "build-vulkan\bin\Release\llama-server.exe"
$defaultModel = "C:\models\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

# --- Config (override via env vars) ---
$model   = if ($env:TQ_MODEL_PATH) { $env:TQ_MODEL_PATH } else { $defaultModel }
$port    = if ($env:TQ_PORT)       { $env:TQ_PORT }       else { "8001" }
$host_ip = if ($env:TQ_HOST)       { $env:TQ_HOST }       else { "127.0.0.1" }
$ctx     = if ($env:TQ_CTX)        { $env:TQ_CTX }        else { "262144" }
$batch   = if ($env:TQ_BATCH)      { $env:TQ_BATCH }      else { "128" }
$ubatch  = if ($env:TQ_UBATCH)     { $env:TQ_UBATCH }     else { "64" }
$ctk     = if ($env:TQ_CTK)        { $env:TQ_CTK }        else { "turbo3" }
$ctv     = if ($env:TQ_CTV)        { $env:TQ_CTV }        else { "turbo3" }
$threads = if ($env:TQ_THREADS)    { $env:TQ_THREADS }    else { "16" }

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

# --- Summary ---
Write-Host "=== Gemma 4 26B-A4B @ 256K (Vulkan + MoE CPU offload) ===" -ForegroundColor Cyan
Write-Host "  model        : $model"
Write-Host "  context      : $ctx tokens"
Write-Host "  KV cache     : K=$ctk V=$ctv (symmetric required for Vulkan FA)"
Write-Host "  batch/ubatch : $batch / $ubatch   (DO NOT raise — compute buffer OOMs)"
Write-Host "  MoE offload  : all 30 expert layers on CPU"
Write-Host "  listen       : http://$host_ip`:$port"
Write-Host ""

# --- Launch ---
& $llamaServer `
    -m $model `
    --alias "gemma-4-26b" `
    -ctk $ctk -ctv $ctv `
    -fa on `
    -ngl 99 `
    -ncmoe 30 `
    -c $ctx `
    -b $batch -ub $ubatch `
    --mlock `
    --no-warmup `
    --threads $threads `
    -np 1 `
    --host $host_ip --port $port `
    @args
