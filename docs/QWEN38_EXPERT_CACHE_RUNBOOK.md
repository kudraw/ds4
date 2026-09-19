# Qwen3.8-Flash-Next expert-cache test runbook

Target: `feature/qwen38-cuda-expert-cache`, built for `sm_120a`.
The cache keeps routed-expert rows in VRAM (LRU, per layer, gate/up/down
slot triples) and reads the RAM-resident mmap'd GGUF only on a miss.
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
per-layer slot size (Q8_0: 3 x 1.5625 MiB = 4.7 MiB; the same slot index
across all 48 layers costs 225 MiB total), cold staging time and effective
GiB/s (PCIe-bound), re-stage time for the same working set (all hits,
microseconds), a shifted working set (fresh misses plus LRU steals on
small budgets) and a `readback OK` verdict, then a stats line. Exit
codes: 0 pass, 1 cache disabled or a check failed, 2 wrong backend, model
or build. This catches geometry mismatches, slab allocation failures or
OOM, and staging corruption before the slower steps below.

### Discrete GPUs (separate VRAM)

On a discrete card the eager full-model tensor preload is a unified-memory
optimization and is **skipped** by default: it would try to `cudaMemcpy` the
entire resident set into VRAM, compete with the expert cache and OOM at load
time. Weights stream from host memory instead, and per-access resolution uses
the per-range zero-copy `cudaHostRegister` path (whole-model registration
fails over PCIe on large files) for whatever is not served from the expert
cache. Look for `CUDA discrete GPU: skipping eager model tensor preload` at
startup. Force the old eager behaviour with `DS4_CUDA_EAGER_PRELOAD_DISCRETE=1`
(for a model that comfortably fits in VRAM). The `resident model ... planned`
memory line is still the unified estimate and overstates VRAM use on discrete.

First green discrete run (RTX PRO 4500 Blackwell, 32 GiB, no explicit budget):

```
ds4: CUDA discrete GPU: skipping eager model tensor preload; weights stream from host, expert cache owns VRAM
ds4: Qwen3.8 routed-expert cache: 18.7 GiB, 79 slab slots per table
qwen4-cache-probe: ngrams q8_0 rows=320001536 width=160 row_bytes=170 read OK (inline, pread only)
qwen4-cache-probe: layer   0  slot 5.0 MiB  slots 79  cold 26.32 ms  re-hit 0.000 ms  shifted 0.00 ms  readback OK
qwen4-cache-probe: layer  24  slot 5.0 MiB  slots 79  cold 27.10 ms  re-hit 0.000 ms  shifted 0.00 ms  readback OK
qwen4-cache-probe: layer  47  slot 5.0 MiB  slots 79  cold 23.48 ms  re-hit 0.000 ms  shifted 0.00 ms  readback OK
[qwen4-expert-cache probe] slots=79/512 hits=60 misses=30 (66.7%) steals=0 staged=0.15 GiB
ds4: expert-cache probe: all checks passed
```

Cold `0.2 GiB/s` is a single cold slot (page fault + first touch + slab alloc),
not the streaming rate. `re-hit 0.000 ms` confirms hits read the slab directly.
`readback OK` proves the publish -> set_window -> slab -> readback round trip is
byte-exact against the model mapping.

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

Default budget first, then explicit values. A slot index carries one expert
across its gate/up/down row per layer, so cost per slot = 144 tables x
1.5625 MiB = 225 MiB: 8 GiB = 37 slots, 16 GiB = 72, 24 GiB = 109 (of 512
experts per layer). Routing picks top-k 10 per layer per token, so a budget
under ~2.3 GiB (10 slots) can never stage a full layer — the stage then
fails every call and the cache is silently inert; keep `MB >= 2304`.

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
