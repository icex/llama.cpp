# RDNA 4 Windows — Gemma 4 26B-A4B

Target hardware: AMD Ryzen 7 9800X3D + 32 GB DDR5 + **RX 9070 XT** (gfx1201, 16 GB) — the product name in some earlier docs was "9700 XT" but the actual retail name is 9070 XT.
Target model: `unsloth/gemma-4-26B-A4B-it-GGUF` at `UD-IQ4_XS` (13.4 GB, ~4.25 bpw).
Target context: 128 K tokens, full GPU, no MoE CPU offload.
Work drives: `E:\work` for the repo, `E:\models` for GGUF files. `C:` is reserved for the OS.

> Earlier versions of this setup used `UD-Q4_K_XL` (17.1 GB) with MoE CPU offload (`-ncmoe 30`) to fit 256 K context. That config works but takes a ~10× prefill / ~4× decode speed hit because experts run on the Zen 5 CPU instead of the 9070 XT. `UD-IQ4_XS` fits natively in 16 GB VRAM without offload, so we lose some quality (4.25 bpw vs 5 bpw) but gain roughly an order of magnitude in prefill throughput. See the bench numbers below.

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

3. Download the model GGUF (17.1 GB) to `E:\models\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` or set `TQ_MODEL_PATH`.

4. Launch the server:
   ```powershell
   .\scripts\rdna4\windows\run-gemma4.ps1
   ```

5. In a second shell, launch Claude Code:
   ```powershell
   .\scripts\rdna4\windows\run-claude-code.ps1
   ```

## Memory budget (why these flags matter)

### GPU VRAM — 16 GB target (measured, b=128 ub=64, q8_0 KV)

| Component | Size | Notes |
|---|---|---|
| Non-expert model tensors (Vulkan0) | ~2.46 GB | attention Q/K/V/O, embeddings, lm_head, norms, MoE routers |
| KV cache global (5 layers × 256K × q8_0) | ~2.72 GB | Gemma 4 K/V use head_dim=128 (only Q uses D=512) |
| KV cache SWA (25 layers × 1280 cells × q8_0) | ~133 MB | fixed window |
| Compute buffer (`-b 128 -ub 64`) | ~338 MB | Vulkan scalar FA is MUCH leaner than Metal. Batch-driven, not ctx-driven |
| Vulkan driver overhead | ~500 MB | |
| **Total** | **~6.2 GB** | **~9.8 GB headroom on 15.4 GB usable** |

> Earlier projection said 12.5 GB with turbo3 KV. Reality on hardware is ~6.2 GB with q8_0 KV because (1) K/V uses head_dim=128 not 512 — only Q is wider in global layers — and (2) Vulkan compute buffer is an order of magnitude smaller than the Metal equivalent.

### System RAM — 32 GB target

| Component | Size | Notes |
|---|---|---|
| Expert REPACK (CPU backend) | ~8.2 GB | Q4_K expert weights reformatted for CPU-efficient access |
| mmap hot working set | ~9 GB | File-backed; OS pages cold experts |
| OS + drivers + desktop | ~5 GB | |
| Claude Code / VSCode / browser | ~3 GB | Close heavy Electron apps during runs |
| **Total active** | **~25 GB** | Tight. `--mlock` keeps experts resident |

## Critical flags — do not change without understanding

- `-b 128 -ub 64`: safe. Raising may eventually blow the compute buffer; Vulkan scalar FA is leaner than Metal so you probably have margin but do not trust the default.
- `-ctk q8_0 -ctv q8_0`: **use q8_0, NOT turbo3**. Vulkan turbo3 is currently broken on Gemma 4 (see "Known issues" below). Symmetric K/V is required by the Vulkan FA path (`op->src[1]->type == op->src[2]->type`), so asymmetric q8_0/turbo3 is also unavailable on Vulkan.
- `-ncmoe 30`: all 30 layers' experts go to CPU. Without this, the 14 GB of expert FFN weights don't fit in 16 GB VRAM.
- `--mlock`: recommended on Windows at 32 GB. Pins expert weights in RAM, prevents pagefile swaps during decode. Requires the `SeLockMemoryPrivilege` right — grant via `gpedit.msc` (Computer Configuration → Windows Settings → Security Settings → Local Policies → User Rights Assignment → "Lock pages in memory") or the script runs without lock (still works, just slower on contention).
- `--no-warmup`: skip empty-batch warmup pass. Safer while sizing.
- `-fit off`: disable the `--fit` auto-tuner. On Gemma 4 the tuner crashes during its probe pass. We size manually.

## Expected performance (projected)

Decode: 25-50 tok/s (CPU MoE bandwidth-bound, DDR5-6000 ~80 GB/s).
Prefill at 256K: 1-3 minutes (first-token latency, cached afterward).

Reference Mac M4 Max same config (full GPU, no CPU MoE offload):
- pp2048: 710 tok/s
- tg64 @ d=0: 49 tok/s
- tg64 @ d=128K: 21 tok/s

The Mac numbers set a ceiling. Windows Vulkan scalar-path performance will be lower than Metal, and CPU MoE adds bandwidth overhead, so expect 40-70 % of Mac decode speed in practice.

## Known issues

### 1. Vulkan turbo3 is broken on Gemma 4 (confirmed)

Two independent bugs prevent `-ctk turbo3 -ctv turbo3` from working on this hardware.

**Bug A: FA pipeline registration gap.** The `CREATE_FA` macro in `ggml/src/ggml-vulkan/ggml-vulkan.cpp` only registers `GGML_TYPE_TURBO3_0` pipelines for the `FA_SCALAR` code path, not `FA_COOPMAT1` or `FA_COOPMAT2` (lines 3451, 3457 — scalar only). The shader generator does compile the cm1/cm2 turbo3_0 SPV binaries, but without matching `CREATE_FA` macro invocations they are never bound to a `vk_pipeline_struct`. On modern AMD/NVIDIA GPUs the FA dispatch defaults to the coopmat2 path, and the lookup of `(turbo3_0, FA_COOPMAT2, ...)` inserts a fresh default-constructed pipeline whose `wg_denoms = {0,0,0}`. The dispatch then asserts `Br == pipeline->wg_denoms[0]` at line 8972 and crashes. Our branch patches `get_fa_tuning_params` to force `FA_SCALAR` for TURBO3_0 (`9def0b36`). Reproduction: RX 9070 XT / Adrenalin 26.3.1 / Vulkan API 1.4.344.

**Bug B: missing kernel-level WHT inverse in Vulkan dequant.** The TurboQuant fork stores K/V in Walsh–Hadamard-rotated basis. Metal and CUDA kernels apply the inverse WHT inline during K/V load (via `simd_shuffle_xor` butterflies) so the downstream FA dot product sees normal-basis values. The Vulkan `dequantize4` function in `flash_attn_base.glsl` only does the naive per-block `centroids[idx] * norm` lookup — no WHT inverse. Attention math runs in the wrong basis → garbled output (Korean/random tokens on an English prompt, confirmed). The upstream graph-level rotation path (`LLAMA_ATTN_ROT_DISABLE=0` enabling `attn_rot_k/v`) cannot fix this either: it's a **separate** rotation (per the code comment at `llama-kv-cache.cpp:426`, "Our fork uses kernel-level WHT rotation... which is independent"), and Gemma 4 is additionally blocked from that path because `is_n_embd_k_gqa_variable()` returns true (SWA head_dim=256 vs global head_dim=512).

**Fix path:** port the turbo3 WHT butterfly (8 subgroup shuffle stages × 128-elem block) from `ggml-metal-impl.h` to a Vulkan shader helper using `subgroupShuffleXor` (VK_KHR_shader_subgroup_shuffle), then call it from `dequantize4` when `DATA_A_TURBO3_0` is defined. Non-trivial but not huge.

**Workaround:** use `-ctk q8_0 -ctv q8_0`. The VRAM cost is modest on Gemma 4 (2.85 GB at 256K vs 1 GB for turbo3) because K/V head_dim is 128, and we have ~9.8 GB of headroom regardless.

### 2. MoE CPU offload on AMD Vulkan is mostly unmeasured

On Mac unified memory `-ncmoe` is a functional-only option: Metal GPU is much faster than the CPU for FFN, so moving experts off the GPU regresses decode speed badly. On Windows the opposite applies — Vulkan cannot hold the experts anyway, CPU is the only option, and Zen 5 + DDR5-6000 + 3D V-Cache is a strong CPU MoE target. Expected: 25-50 tok/s decode at small context, degrading with KV depth. Measured numbers go in `bench-results.md` (TBD).

### 3. 32 GB RAM is tight

Active working set projects to ~25 GB. Close VSCode, Chrome, Slack, etc. before runs. A desktop with ~8 GB wired for OS leaves you at 24 GB for the inference process.

## Troubleshooting

- **OOM at load time**: check `-b / -ub`. Default is way too aggressive.
- **OOM during warmup**: add `--no-warmup` (already on by default in our script).
- **Garbled output at turbo3**: verify with `-ctk q8_0 -ctv q8_0` and smaller context. If q8_0 works and turbo3 doesn't, the Vulkan D=256/512 turbo3 path has a correctness issue — report upstream.
- **Slow decode (< 10 tok/s)**: check `--mlock` took effect (Task Manager → Performance → Memory → Committed). If pagefile is growing, `--mlock` is silently failing due to quota; grant "Lock pages in memory" user right via `gpedit.msc` → Computer Configuration → Windows Settings → Security Settings → Local Policies → User Rights Assignment.
- **Claude Code KV cache thrashing**: verify `CLAUDE_CODE_ATTRIBUTION_HEADER: "0"` is in `%USERPROFILE%\.claude\settings.json`. Without this, decode speed drops ~90 %.
