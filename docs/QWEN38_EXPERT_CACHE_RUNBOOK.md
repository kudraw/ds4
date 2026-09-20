# Qwen3.8-Flash-Next expert-cache test runbook

Target: `feature/qwen38-cuda-expert-cache`, built for `sm_120a`.
Test hardware: NVIDIA RTX PRO 4500 Blackwell, 32 GiB GDDR7 ECC, 896 GB/s,
PCIe Gen 5 (`sm_120`). Default budget = 60% of VRAM free after the engine
loads. The cache keeps routed-expert rows in **3 shared VRAM slabs** (gate /
up / down), one slot pool reused by every routed layer (LRU, triple-tagged),
and reads the RAM-resident mmap'd GGUF only on a miss. The pool is **not**
capped at the 512-expert count: routed experts reach their slab row through a
slot **sentinel** (`NE + slot`, `NE` = 512) carried in the expert-id field, a
band disjoint from raw ids `[0, NE)`, so `slot_count = budget / per-slot-bytes`
grows **linearly with the budget**. That lets the resident working set span
**multiple layers** — a single layer routes ≤512 distinct experts but 48 layers
route far more — which is what lifts the cross-layer hit rate. Per-slot cost =
gate+up+down ≈ 4.98 MiB (Q8_0), so a 24 GiB budget yields ~4934 slots. Size the
budget so `slabs + dense (~5 GiB) + KV/buffers (~1.2 GiB)` still fits VRAM; a
corrupt id past `NE + slot_count` reads out-of-bounds and trips a CUDA error
(loud) rather than silently misreading.
Toggles:

- `DS4_QWEN4_EXPERT_CACHE_MB=N` — VRAM budget for expert slabs; `0` disables
  the cache (kernels use raw ids); unset = 60% of free VRAM at first forward.
- `DS4_QWEN4_EXPERT_CACHE_STATS=1` — one summary line on stderr at exit:
  slots, hits/misses %, steals, staged GiB.

Runtime needs `LD_LIBRARY_PATH=/opt/cuda/lib64` unless the CUDA runtime is
registered with ldconfig.

## 1. Build

```sh
make -j8 CUDA_ARCH=sm_120a ds4 tests/ds4_qwen4_expert_cache_test
```

## 2. Unit smoke test (few MB of VRAM, safe next to a running server)

```sh
./tests/ds4_qwen4_expert_cache_test
```

Expect a `PASS` line per check and `ALL PASS 0 failure(s)`; exit code 0.
Any `FAIL` line + the surrounding context is the report I need.

## 3. Merge the shards

The engine mmaps a single file at offset 0 and has no `split.*` handling, so
merge the 10 shards first (~165.7 GiB output; check free disk):

```sh
/path/llama.cpp/build/bin/gguf-split --merge \
  /path/Qwen3.8-Flash-Next-Q8_0.gguf-00001-of-00010.gguf \
  /path/Qwen3.8-Flash-Next-Q8_0-merged.gguf
```

Sanity-check the result with the python `gguf` package:

```python
from gguf import GGUFReader
r = GGUFReader("/path/Qwen3.8-Flash-Next-Q8_0-merged.gguf")
assert len(r.tensors) == 1224, len(r.tensors)
assert not any(k.startswith("split.") for k in r.fields)
```

## 4. Engine probe (machine free, no generation)

`--qwen4-cache-probe` opens the engine with the production cache
configuration path, stages synthetic routing working sets on selected
layers, copies staged rows back from VRAM and compares them against the
model mapping byte for byte. No prompt, no sampling; exit code 0 = pass:

```sh
DS4_QWEN4_EXPERT_CACHE_MB=14336 ./ds4 --cuda -m /path/...-merged.gguf \
  --qwen4-cache-probe
```

Default layers are first/middle/last; select explicitly with
`--qwen4-cache-probe-layers 0,1,46,47`. Each layer prints one line:
per-layer slot size (Q8_0: 3 x 1.66 MiB = 4.98 MiB; the 3 slabs are **shared**
and the probe opens one window at a time, so the same slot index is reused
across all 48 layers — cross-layer reuse costs PCIe re-stage traffic, not
extra VRAM), cold staging time and effective
GiB/s (PCIe-bound), re-stage time for the same working set (all hits,
microseconds), a shifted working set (fresh misses plus LRU steals on
small budgets) and a `readback OK` verdict, then a stats line. Exit
codes: 0 pass, 1 cache disabled or a check failed, 2 wrong backend, model
or build. This catches geometry mismatches, slab allocation failures or
OOM, and staging corruption before the slower steps below.

### Discrete GPUs (separate VRAM)

On a discrete card VRAM is a separate, scarce pool shared by the expert cache,
the KV/activation working set and the dense weights. The eager preload is
split by role:

- **Dense / always-on weights** (attention, embeddings, layernorm, router,
  shared experts `*_shexp.weight`) are **eager-copied into VRAM once** at load,
  so kernels read them from device memory. This replaces the per-access
  per-range `cudaHostRegister` path that fails over PCIe (invalid argument on a
  slice of the file-backed mmap) and then `cudaMalloc`s per tensor until it OOMs
  at token 0.
- **Routed experts** (`*_exps.weight`) are **skipped by the eager arena** and keep
  streaming through the expert cache. That path does `cudaMemcpy` H2D straight
  from the pageable mmap into VRAM slabs and needs **no host pinning**, so it is
  unaffected by the whole-file `cudaHostRegister` limit.

If the dense set alone does not fit VRAM a span failure is non-fatal (warns
once, streams the rest) instead of aborting startup. A whole-model pin whose
span exceeds host RAM is skipped (it can never page-lock and only thrashes the
page cache the streaming path relies on). All of this is gated on a discrete
detection — the unified/GB300 arena and the routed path are unchanged. Force
preload-everything with `DS4_CUDA_EAGER_PRELOAD_DISCRETE=1` (only sane when the
whole model fits VRAM). Look for `CUDA discrete GPU: eager-preloading
dense/non-routed weights only` at startup. The `resident model ... planned`
memory line is still the unified estimate and overstates VRAM use on discrete.

First green discrete generation run under the **sentinel (uncapped) pool**
(RTX PRO 4500 Blackwell, 32 GiB, budget `24576`):

```
ds4: CUDA discrete GPU: eager-preloading dense/non-routed weights only; routed experts stream from host via the expert cache
ds4: CUDA startup model preparation covered 5.09 GiB of tensor spans in 0.843s
ds4: expert slabs: 3 shared pools (gate/up/down) x 4934 slots (cudaMalloc VRAM, reused across all 144 routed tables; rows H2D-copied in from the model on window stage; per-slot gate/up/down stride 1.66/1.66/1.66 MiB), total 24568 MiB resident
ds4: Qwen3.8 routed-expert cache: budget 24.0 GiB -> 4934 slots
ds4: Qwen MoE: routed-expert cache active -> using STREAMING per-token path (tiled-GEMM/mm disabled)
ds4: prefill: 22.45 t/s, generation: 22.44 t/s
[qwen4-expert-cache final] slots=4934/512 hits=221806 misses=46034 (82.8%) steals=41100 staged=223.90 GiB
```

Budget `24576` -> **4934 slots** (no 512 cap; `slots` in the stats line is
`slot_count/slot`, so `4934/512`), ~24 GiB resident. Dense weights sit in 5.09
GiB of device copies, KV+buffers ~1.23 GiB: the whole footprint fits 32 GiB.
The sentinel un-cap (Design A) is the reason generation jumped from the 512-slot
figures (~7.9 t/s at the old cap) to **22.44 t/s** here: with `slot_count`
holding the working set of many layers at once, the global LRU keeps rows warm
across layers (82.8% hit) instead of re-staging every layer's experts each
token. `staged=223.90 GiB` over 267840 lookups against a 24 GiB pool is the
residual PCIe traffic the 41100 steals account for.

Earlier probe evidence (32 GiB, no explicit budget) still holds for the
byte-exactness round trip:

```
qwen4-cache-probe: ngrams q8_0 rows=320001536 width=160 row_bytes=170 read OK (inline, pread only)
qwen4-cache-probe: layer   0  slot 5.0 MiB  cold 26.32 ms  re-hit 0.000 ms  shifted 0.00 ms  readback OK
```

`re-hit 0.000 ms` confirms hits read the slab directly. `readback OK` proves
the publish -> set_window -> slab -> readback round trip is byte-exact against
the model mapping.

## 5. Correctness A/B (machine free, GPU idle)

Same prompt, cache off vs on. The cache is a pure optimization — outputs
must match exactly:

```sh
DS4_QWEN4_EXPERT_CACHE_MB=0 ./ds4 --cuda -m /path/...-merged.gguf -c 4096 \
  -p "Explain why MoE models route tokens to experts." > off.txt
DS4_QWEN4_EXPERT_CACHE_MB=4096 DS4_QWEN4_EXPERT_CACHE_STATS=1 ./ds4 --cuda \
  -m /path/...-merged.gguf -c 4096 \
  -p "Explain why MoE models route tokens to experts." > on.txt
diff off.txt on.txt && echo IDENTICAL
```

Then repeat the second command (fresh process, warm cache) and again after a
second identical prompt in the same session — hit rates should climb.

## 6. Performance sweep (machine free)

Default budget first, then explicit values. The 3 slabs form **one shared pool**
(gate/up/down), reused by all 144 routed tables; cost per slot = gate+up+down
≈ 4.98 MiB (Q8_0). With the sentinel pool `slot_count` scales **linearly with
the budget** (no 512 cap): `MB / 4.98 MiB ≈ slots` (24576 -> 4934). The larger
the resident pool, the more layers' working sets stay warm at once and the
higher the hit rate — generation tracked `slot_count` monotonically in testing
(512-slot ~7.9 t/s -> 4934-slot 22.44 t/s). Full residency of every routed
expert (144 tables × 512 ≈ 359 GiB) is not reachable on 32 GiB, so pick the
budget by VRAM headroom: leave room for dense (~5 GiB) + KV/buffers (~1.2 GiB).
Routing picks top-k 10 per layer per token, so a budget under ~50 MiB (<10
slots) can never stage a full layer's working set — the stage reports failure
with a one-time warning (host map stays consistent with the window); keep the
budget well above that. Low budgets trade resident slots for LRU re-stage
(steal) traffic.

```sh
for MB in 0 8192 16384 24576; do
  DS4_QWEN4_EXPERT_CACHE_MB=$MB DS4_QWEN4_EXPERT_CACHE_STATS=1 \
    ./ds4 --cuda -m /path/...-merged.gguf -c 8192 -p "$(python3 -c \
    'print("Summarize the architecture of a 512-expert MoE. "*200)')" \
    2>&1 | tail -5
done
```

If the slab allocation fails at a high budget (graph/KV needs the rest of
VRAM), lower it; the error message names the failed allocation.

## What to send back

1. Unit test output (step 2), 2. the probe output and exit code (step 4),
3. `diff` result from step 5, 4. the `[qwen4-expert-cache final]` lines +
tokens/s from step 6, 5. any allocation errors with the exact text.
