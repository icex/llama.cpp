#!/usr/bin/env pwsh
# Start Claude Code CLI against the local llama-server with Gemma 4.
# Assumes run-gemma4.ps1 is already running on the configured port.

$ErrorActionPreference = "Stop"

$port    = if ($env:TQ_PORT) { $env:TQ_PORT } else { "8001" }
$host_ip = if ($env:TQ_HOST) { $env:TQ_HOST } else { "127.0.0.1" }
$server  = "http://$host_ip`:$port"

# --- Sanity check server is up ---
try {
    $null = Invoke-WebRequest -Uri "$server/health" -TimeoutSec 2 -UseBasicParsing
    Write-Host "llama-server reachable at $server" -ForegroundColor Green
} catch {
    Write-Host "Cannot reach llama-server at $server" -ForegroundColor Red
    Write-Host "Start it first: .\scripts\rdna4\windows\run-gemma4.ps1" -ForegroundColor Yellow
    exit 1
}

# --- Client env ---
$env:ANTHROPIC_BASE_URL  = $server
$env:ANTHROPIC_AUTH_TOKEN = "local"
$env:ANTHROPIC_API_KEY   = ""
$env:ANTHROPIC_MODEL     = "gemma-4-26b"

# --- KV cache preservation ---
# Claude Code adds an attribution header by default. Every request with a new
# header permutation invalidates the prompt cache on the server side. Setting
# this to 0 prevents a ~90% decode-speed regression.
# (Also needs CLAUDE_CODE_ATTRIBUTION_HEADER=0 in %USERPROFILE%\.claude\settings.json.)
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"

$settingsPath = Join-Path $env:USERPROFILE ".claude\settings.json"
if (-not (Test-Path $settingsPath)) {
    Write-Host "WARNING: $settingsPath missing." -ForegroundColor Yellow
    Write-Host "  Create it with: { `"env`": { `"CLAUDE_CODE_ATTRIBUTION_HEADER`": `"0`" } }" -ForegroundColor Yellow
} else {
    $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
    $attrib = $settings.env.CLAUDE_CODE_ATTRIBUTION_HEADER
    if ($attrib -ne "0") {
        Write-Host "WARNING: CLAUDE_CODE_ATTRIBUTION_HEADER not set to '0' in settings.json" -ForegroundColor Yellow
        Write-Host "  Without this, Claude Code invalidates the KV cache on every request." -ForegroundColor Yellow
    }
}

# --- Launch ---
Write-Host "Launching Claude Code..." -ForegroundColor Cyan
npx -y "@anthropic-ai/claude-code@latest" @args
