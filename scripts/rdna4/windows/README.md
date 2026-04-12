# RDNA 4 Windows — Gemma 4 26B-A4B @ 256K

Target hardware: AMD Ryzen 7 9800X3D + 32 GB DDR5 + RX 9700 XT (gfx1201, 16 GB).
Target model: `unsloth/gemma-4-26B-A4B-it-GGUF` at `UD-Q4_K_XL` (17.1 GB).
Target context: 256 K tokens.

## Bring-up

1. Install prerequisites (one-time):
   - **Visual Studio 2022 Build Tools** with "Desktop development with C++" workload
   - **CMake ≥ 3.25**: `winget install Kitware.CMake`
   - **Git**: `winget install Git.Git`
   - **Vulkan SDK**: https://vulkan.lunarg.com (installer sets `VULKAN_SDK`)
   - **AMD Adrenalin 25.x** driver: https://amd.com/support
   - **Node.js LTS** (for Claude Code): `winget install OpenJS.NodeJS.LTS`

   Reboot after the driver install.

2. Open **"x64 Native Tools Command Prompt for VS 2022"**, launch PowerShell inside it (`pwsh` or `powershell`), then:

   ```powershell
   cd path\to\llama-cpp-turboquant
   .\scripts\rdna4\windows\check-prereqs.ps1   # verifies tools, driver, GPU, RAM, model
   .\scripts\rdna4\windows\build.ps1           # configures + builds Vulkan backend
   ```

3. Download the model GGUF (17.1 GB) to `C:\models\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` or set `TQ_MODEL_PATH`.

4. Launch the server:
   ```powershell
   .\scripts\rdna4\windows\run-gemma4-256k.ps1
   ```

5. In a second shell, launch Claude Code:
   ```powershell
   .\scripts\rdna4\windows\run-claude-code.ps1
   ```

## Memory budget (why these flags matter)

### GPU VRAM — 16 GB target

| Component | Size | Notes |
|---|---|---|
| Non-expert model tensors | ~2.7 GB | attention Q/K/V/O, embeddings, lm_head, norms, MoE routers |
| KV cache turbo3 @ 256K | ~1.05 GB | 1.0 GB global (5 layers × 256K × D=512) + 49 MB SWA (25 layers × 1280 cells) |
| Compute buffer (`-b 128 -ub 64`) | ~8.3 GB | Batch-driven; does NOT scale with context. Default `-b 2048` allocates ~32 GB and instant-OOMs |
| Vulkan driver overhead | ~0.5 GB | |
| **Total** | **~12.5 GB** | ~3.5 GB headroom |

### System RAM — 32 GB target

| Component | Size | Notes |
|---|---|---|
| Expert REPACK (CPU backend) | ~8.2 GB | Q4_K expert weights reformatted for CPU-efficient access |
| mmap hot working set | ~9 GB | File-backed; OS pages cold experts |
| OS + drivers + desktop | ~5 GB | |
| Claude Code / VSCode / browser | ~3 GB | Close heavy Electron apps during runs |
| **Total active** | **~25 GB** | Tight. `--mlock` keeps experts resident |

## Critical flags — do not change without understanding

- `-b 128 -ub 64`: dropping below is fine, raising breaks VRAM budget. At `-b 256` compute buffer grows to ~16 GB (OOM).
- `-ctk turbo3 -ctv turbo3`: Vulkan FA path rejects mixed K/V types (`op->src[1]->type != op->src[2]->type` → false). You cannot use asymmetric q8_0/turbo3 on Vulkan — that only works on Metal/CUDA.
- `-ncmoe 30`: all 30 layers' experts go to CPU. Without this, the 14 GB of expert FFN weights force-push model into VRAM and OOM.
- `--mlock`: mandatory on Windows at 32 GB. Prevents pagefile swaps of expert weights. Without it, a single page fault during decode can stall the token by seconds.
- `--no-warmup`: prevents an empty-batch allocation spike that can OOM at tight VRAM.

## Expected performance (projected)

Decode: 25-50 tok/s (CPU MoE bandwidth-bound, DDR5-6000 ~80 GB/s).
Prefill at 256K: 1-3 minutes (first-token latency, cached afterward).

Reference Mac M4 Max same config (full GPU, no CPU MoE offload):
- pp2048: 710 tok/s
- tg64 @ d=0: 49 tok/s
- tg64 @ d=128K: 21 tok/s

The Mac numbers set a ceiling. Windows Vulkan scalar-path performance will be lower than Metal, and CPU MoE adds bandwidth overhead, so expect 40-70 % of Mac decode speed in practice.

## Known risks

1. **Vulkan turbo3 at D=256 / D=512 is untested on Gemma 4.** The code paths exist (`flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp` all compile turbo3_0 pipeline variants via `DATA_A_TURBO3_0`; FA dispatch accepts `GGML_TYPE_TURBO3_0` for any `HSK % 8 == 0`). Metal was explicitly validated by commit `716dd77`. Vulkan was not. First smoke test after build should verify generation quality; garbled output means fall back to q8_0 and accept smaller context.

2. **MoE CPU offload on AMD Vulkan is unmeasured.** The `-ncmoe` flag is backend-agnostic in theory. On Mac it works functionally but regresses speed badly because Metal GPU > Mac CPU for FFN. On Windows the opposite should hold — Vulkan GPU cannot hold the experts anyway, so CPU is the only option, and Zen 5 + DDR5 + 3D V-Cache is a strong CPU MoE target.

3. **32 GB RAM is tight.** Active working set projects to ~25 GB. Close VSCode, Chrome, Slack, etc. before runs. A desktop with ~8 GB wired for OS leaves you at 24 GB for the inference process.

## Troubleshooting

- **OOM at load time**: check `-b / -ub`. Default is way too aggressive.
- **OOM during warmup**: add `--no-warmup` (already on by default in our script).
- **Garbled output at turbo3**: verify with `-ctk q8_0 -ctv q8_0` and smaller context. If q8_0 works and turbo3 doesn't, the Vulkan D=256/512 turbo3 path has a correctness issue — report upstream.
- **Slow decode (< 10 tok/s)**: check `--mlock` took effect (Task Manager → Performance → Memory → Committed). If pagefile is growing, `--mlock` is silently failing due to quota; grant "Lock pages in memory" user right via `gpedit.msc` → Computer Configuration → Windows Settings → Security Settings → Local Policies → User Rights Assignment.
- **Claude Code KV cache thrashing**: verify `CLAUDE_CODE_ATTRIBUTION_HEADER: "0"` is in `%USERPROFILE%\.claude\settings.json`. Without this, decode speed drops ~90 %.
