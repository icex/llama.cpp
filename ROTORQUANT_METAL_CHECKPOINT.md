# Rotorquant Metal Optimization — Checkpoint

**Branch**: `wip/rotorquant-metal-checkpoint-20260415`
**Base**: `feature/planarquant-kv-cache`
**Date**: 2026-04-15
**Status**: WIP — paused because the micro-optimization approach isn't scaling.

## Why this branch exists

Gemma 4 wouldn't load on rotorquant's `feature/planarquant-kv-cache` at all. Once
ported, decode ran into structural perf limits (86 graph splits from missing
`dk=512` Metal FA templates) that silently fell back to CPU. This branch fixes
the structural bugs and ports the cheap optimizations already validated in
turboquant. It's the minimum viable functional state on Metal M4 Max for
Gemma 4 E4B.

It does **not** contain the bigger architectural changes that would close the
remaining gap to f16 decode speed. See "Why we stopped" below.

## What's in this branch (8 commits on top of base)

| Commit | Effect on Gemma 4 E4B M4 Max |
|---|---|
| `fbd2d4e75` Gemma 4 arch port | model loads |
| `b422a54f7` dk=512 FA templates + deferred-K CUDA guard | 86 splits → 2, prefill 390 → 1175 t/s |
| `a149d568e` sparse V auto-enable (not just M5+) | iso3 decode at 32k: 32.5 → 39.3 t/s (+21%) |
| `cf47b5acb` TurboFlash decode kernel | turbo3 @ 32k +37% (head_dim ≤ 128 only) |
| `bd45fc0be` Q8_0 × iso3/planar3 asymmetric FA templates | q8_0/iso3 decode +15% vs sym iso3 |
| `e68df384e` F16 × iso3 investigation notes | known bug, path disabled |
| `a48f2c957` Cleanup after F16 wrapper attempt | — |
| `a8dd45624` Vectorized Hamilton + tunable TURBO_SPARSE_V_THRESHOLD | neutral (compiler already vectorized) |

## Final cold-bench numbers (Gemma 4 E4B Q4_K_M, M4 Max, `-fa on -p 512 -n 128 -r 3`)

| Config | pp512 | tg128 | Compression |
|---|---|---|---|
| f16 / f16 | 1197 | 80.6 | 1x |
| q8_0 / q8_0 | 1182 | 72.4 | 2x |
| **q8_0 / iso3** (recommended) | **1176** | **59.5** | **~4.2x total** |
| planar3 / planar3 | 1178 | 52.8 | 5.1x |
| iso3 / iso3 | 1173 | 52.2 | 5.1x |
| turbo3 / turbo3 | 1167 | 50.5 | 5.1x |

## Why we stopped

**1. The decode gap between iso3 (52 tg) and q8_0 (72 tg) can't be closed by
micro-optimizing the dequant kernel.** Profiled by stripping:

| State | tg128 | Δ vs full |
|---|---|---|
| Full dequant | 52.1 | baseline |
| Rotation stripped | 53.9 | +3% |
| Centroid LUT stripped | 58.1 | +12% |
| Both stripped | 58.4 | +12% |

The measured ceiling (everything stripped) is only 58 tg — still 14 tg below
q8_0. And three LUT-reduction attempts (4-entry mag + XOR sign, branchless
`select` chains, packed-half4 constant) **all neutral or slightly slower**.
The MSL compiler is already doing the right thing on this code; there's no
low-hanging ALU or cache win left.

**2. The `f16 × iso3/planar3` FA kernel has a latent correctness bug in a code
path upstream llama.cpp never exercises** (neither rotorquant nor turboquant
has any `kf16_v<quant>` templates). Both direct and wrapper-type approaches
produce corrupted decode output despite dispatching cleanly to Metal.

**3. The "40× performance bump" premise was wrong.** Re-reading the rotorquant
README, the advertised gains are 5.3× prefill and 28% decode, both for
`planar3 vs turbo3 on CUDA RTX 5090`. Both sit **below** the CUDA f16
baseline (planar3 is 62% of f16 prefill; the 5.3× gap exists only because
CUDA's turbo3 prefill is shockingly slow at 722 t/s vs f16 6156). On Metal
M4 Max after the fixes in this branch, **all three rotation types already
match the f16 prefill ceiling within 2%**. There's no equivalent gap to
close.

## What a "better approach" would look like (for next session)

The current micro-optimization approach is capped. Higher-leverage directions:

### Option A: Accept q8_0 × iso3 as the answer, ship it

Don't chase iso3/iso3 decode further. Document `-ctk q8_0 -ctv iso3` as the
canonical "fast + compressed" config on Metal and invest effort elsewhere
(e.g. MLX Swift path, speculative decoding, different model).

Effort: 0. Gain: already delivered (+15% decode over sym iso3, ~4.2× KV
compression, f16-quality K precision).

### Option B: Replace iso3/planar3 with a fundamentally different dequant structure

Instead of "3-bit centroid index + separate signs byte + per-group rotation",
try:

- **Stored-dequantized approach**: pre-compute `centroid × rotation` at
  `set_rows` time and store the resulting half value directly. Costs more
  memory (closer to f16) but eliminates the dequant hot path entirely.
  Breaks compression but may be interesting for a "fast iso3" variant.

- **Smaller block size** (32 elements instead of 128): reduces per-block
  overhead, enables simdgroup-wide block loading. q8_0 at QK=32 is the
  reference shape that the Metal FA dispatch assumes — it's the reason q8_0
  decodes faster than iso3 (QK=128) despite having similar math.

- **Fused rotation + attention kernel** (like TurboFlash for turbo3 but
  adapted for per-group quaternion/Givens). 200-500 lines of specialized
  Metal, potentially ~20% decode gain on iso3. This is the direction most
  likely to move the needle.

### Option C: Move rotation to Q side (new ggml op)

Implement `ggml_iso_rotate_q` / `ggml_planar_rotate_q` that applies the per-
group quaternion rotation to Q before attention. Dequant then reduces to
`centroid × norm` (no rotation math). Analogous to how turbo3 uses
`ggml_turbo_wht`.

Effort: ~200-300 lines. Gain: ~3% decode (the measured rotation cost). Not
worth it alone, but nearly free if already writing a new op for Option B.

### Option D: Port upstream turboquant decode optimizations as they land

Rotorquant is ~2 weeks behind turboquant's mainline. A quarterly catch-up
cherry-pick absorbs unrelated wins cheaply.

Effort: 1-2 hours per quarter. Gain: whatever turboquant adds.

## Pickup notes

When you come back to this:

1. **First**: re-run the cold bench on your current hardware to confirm the
   numbers in this document still hold. Thermal state matters a lot on M4 Max
   — use `-r 5` and cool-down sleeps between runs.

2. **Decide on Option A vs B**: is the current q8_0 × iso3 good enough for
   production, or is there a concrete 256k-context use case where you need
   the symmetric iso3/iso3 path to be faster?

3. **If pursuing Option B**: start with the **stored-dequantized half value**
   experiment since it's the smallest test of whether removing the LUT
   actually gives a win (vs the "both stripped → 58 tg" ceiling we measured).
   If that experiment caps at 58 tg too, the problem is elsewhere (bandwidth
   from block struct layout, or kernel register pressure).

4. **F16 × iso3 debugging**: if it becomes worth pursuing, use Xcode Metal
   capture on a minimal test case. Key diagnostic: compare the `ss[]`
   attention score buffer contents between the working `iso3 × iso3` and
   broken `f16 × iso3` cases at the same input. They should be bit-identical.
   If they differ, the bug is on the K path. If identical, the bug is on the
   V path.

5. **Reference sibling branches**:
   - `port/gemma4-v2` — same content as this checkpoint, kept as a shortcut
   - `feature/planarquant-kv-cache` — upstream rotorquant
   - `turboquant/integration/macos-gemma4-128k-20260412` — sibling fork with
     Gemma 4 + more optimizations (different codebase history)

## Evaluation report

Full writeup with bench methodology, diagnostic traces, and all the dead-end
attempts at `/Users/bogdan/Sites/turboquant_plus/docs/rotorquant-evaluation.md`.
