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

## 4. Correctness A/B (machine free, GPU idle)

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

## 5. Performance sweep (machine free)

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

1. Unit test output (step 2), 2. `diff` result from step 4,
3. the `[qwen4-expert-cache final]` lines + tokens/s from step 5,
4. any allocation errors with the exact text.
