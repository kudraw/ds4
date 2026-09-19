/*
 * Unit smoke test for the Qwen3.8-Flash-Next routed-expert VRAM cache
 * (ds4_qwen4_expert_cache).  Synthetic tables, no GGUF required: exercises
 * configure/slot math, LRU staging, cross-layer steals, the too-many-ids
 * guard, teardown, and that window publication succeeds against the CUDA
 * backend.  Prints one "PASS"/"FAIL ..." line per check and a summary;
 * exit code 0 only when everything passed.
 *
 * Build:  make CUDA_ARCH=sm_120a tests/ds4_qwen4_expert_cache_test
 * Run:    ./tests/ds4_qwen4_expert_cache_test        (needs a CUDA device)
 */
#include "ds4_gpu.h"
#include "ds4_qwen4_expert_cache.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_TABLES 6u /* 2 layers x (gate, up, down) */
#define N_LAYERS (N_TABLES / 3u)
#define COUNT 32u   /* experts per table */
#define ROW 4096ull /* synthetic expert row bytes */
#define SLOTS 8u    /* slots per table at the budget below */

static int g_fail;

static void check(int ok, const char *what) {
    if (ok) {
        printf("PASS %s\n", what);
    } else {
        printf("FAIL %s\n", what);
        g_fail++;
    }
}

static void check_u(unsigned got, unsigned want, const char *what) {
    if (got == want) {
        printf("PASS %s (%u)\n", what, got);
    } else {
        printf("FAIL %s: got %u want %u\n", what, got, want);
        g_fail++;
    }
}

static int32_t lookup_or_neg(uint32_t layer, uint32_t expert) {
    int32_t slot = -2;
    if (!ds4_qwen4_expert_cache_lookup(layer, expert, &slot))
        return -2;
    return slot;
}

/* Byte pattern identifying (table, expert) in the synthetic model map. */
static void fill_map(unsigned char *map) {
    for (uint32_t t = 0; t < N_TABLES; t++) {
        for (uint32_t e = 0; e < COUNT; e++) {
            unsigned char *row = map + (uint64_t)t * COUNT * ROW + (uint64_t)e * ROW;
            const unsigned char tag = (unsigned char)(t * 64u + e);
            for (uint64_t i = 0; i < ROW; i++)
                row[i] = (unsigned char)(tag ^ (i & 0xff));
        }
    }
}

int main(void) {
    printf("qwen4-expert-cache unit test: tables=%u layers=%u count=%u row=%llu\n",
           N_TABLES, N_LAYERS, COUNT, (unsigned long long)ROW);
    /* ds4_gpu_init() returns 1 on success, 0 on failure (same convention
     * as tests/cuda_long_context_smoke.c).  Failure reasons are printed by
     * the backend itself as "ds4: ..." lines on stderr. */
    const int gpu_rc = ds4_gpu_init();
    if (gpu_rc <= 0) {
        printf("FAIL ds4_gpu_init returned %d (expected 1; backend prints the "
               "reason on stderr - check the device is visible: nvidia-smi, "
               "CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH for libcudart)\n", gpu_rc);
        return 2;
    }
    printf("PASS ds4_gpu_init (rc=%d)\n", gpu_rc);

    unsigned char *map = malloc((size_t)(N_TABLES * COUNT * ROW));
    if (!map) return 2;
    fill_map(map);

    ds4_qwen4_expert_table tables[N_TABLES];
    for (uint32_t t = 0; t < N_TABLES; t++) {
        tables[t].file_offset = (uint64_t)t * COUNT * ROW;
        tables[t].bytes = COUNT * ROW;
        tables[t].count = COUNT;
        tables[t].type = 8; /* Q8_0 */
    }

    /* 1. Budget below one triple must refuse; exact budget must fit. */
    check(!ds4_qwen4_expert_cache_configure(tables, N_TABLES, map, N_TABLES * ROW - 1),
          "configure refuses budget smaller than one slot");
    const uint64_t budget = (uint64_t)SLOTS * N_TABLES * ROW + 123;
    check(ds4_qwen4_expert_cache_configure(tables, N_TABLES, map, budget), "configure succeeds");
    check(ds4_qwen4_expert_cache_enabled(), "enabled after configure");
    check_u(ds4_qwen4_expert_cache_slot_count(), SLOTS, "slot count = budget/(tables*row)");

    int32_t slots[COUNT];
    const int32_t ids_a[] = {1, 2, 3};

    /* 2. First stage loads all three; slots distinct and in range. */
    check(ds4_qwen4_expert_cache_stage(0, ids_a, 3, slots), "stage layer0 loads 3 ids");
    int ok = 1;
    for (uint32_t i = 0; i < 3; i++)
        ok &= slots[ids_a[i]] >= 0 && (uint32_t)slots[ids_a[i]] < SLOTS;
    ok &= slots[10] == -1 && slots[0] == -1;
    int distinct = slots[1] != slots[2] && slots[2] != slots[3];
    check(ok && distinct, "staged ids have distinct in-range slots; others -1");

    /* 3. Restaging the same ids is a hit: same slots, nothing stolen. */
    check(ds4_qwen4_expert_cache_stage(0, ids_a, 3, slots), "restage same ids succeeds");
    ok = lookup_or_neg(0, 1) >= 0 && lookup_or_neg(0, 2) >= 0 && lookup_or_neg(0, 3) >= 0;
    check(ok, "restaged ids still resident");

    /* 4. Overfill one layer: free slots fill, then the stage must fail
     *    (never evict a slot already published by the same call). */
    int32_t many[COUNT];
    for (uint32_t i = 0; i < COUNT; i++)
        many[i] = (int32_t)i;
    check(!ds4_qwen4_expert_cache_stage(0, many, COUNT, slots),
          "stage of more distinct ids than slots fails");

    /* 5. Cross-layer stage: layer1 asks for exactly SLOTS ids.  Layer0's
     *    tagged slots are stealable, so all 8 load (3 stolen + 5 free);
     *    layer0 is evicted.  A second same-sized layer1 batch must then
     *    fail: every slot now carries layer1's tag and none may be stolen
     *    while published. */
    int32_t layer1_ids[SLOTS];
    for (uint32_t i = 0; i < SLOTS; i++)
        layer1_ids[i] = (int32_t)(16 + i);
    check(ds4_qwen4_expert_cache_stage(1, layer1_ids, SLOTS, slots),
          "stage layer1 steals layer0 slots and loads all");
    ok = 1;
    for (uint32_t i = 0; i < SLOTS; i++)
        ok &= lookup_or_neg(1, (uint32_t)layer1_ids[i]) >= 0;
    check(ok, "all layer1 ids resident");
    ok = 1;
    for (uint32_t e = 0; e < COUNT; e++)
        ok &= lookup_or_neg(0, e) == -1;
    check(ok, "layer0 fully evicted by layer1 steals");
    /* Same-layer re-stage with fresh ids: global LRU evicts the coldest
     * rows (layer1's own previous batch) and succeeds; only slots already
     * published by the current call are untouchable. */
    int32_t layer1_ids2[SLOTS];
    for (uint32_t i = 0; i < SLOTS; i++)
        layer1_ids2[i] = (int32_t)(i); /* fresh ids, same layer */
    check(ds4_qwen4_expert_cache_stage(1, layer1_ids2, SLOTS, slots),
          "layer1 re-stage evicts its own cold rows (global LRU)");
    ok = 1;
    for (uint32_t i = 0; i < SLOTS; i++) {
        ok &= lookup_or_neg(1, (uint32_t)layer1_ids2[i]) >= 0;
        ok &= lookup_or_neg(1, (uint32_t)(16 + i)) == -1;
    }
    check(ok, "re-staged ids resident, previous batch evicted");

    /* 6. Unstage is idempotent and stage after teardown fails. */
    ds4_qwen4_expert_cache_unstage();
    ds4_qwen4_expert_cache_unstage();
    check(1, "unstage idempotent");
    ds4_qwen4_expert_cache_shutdown();
    check(!ds4_qwen4_expert_cache_enabled(), "disabled after shutdown");
    check(!ds4_qwen4_expert_cache_stage(0, ids_a, 3, slots), "stage fails after shutdown");
    ds4_qwen4_expert_cache_shutdown();
    check(1, "double shutdown safe");

    /* 7. Reconfigure with a huge budget clamps slots to the expert count. */
    check(ds4_qwen4_expert_cache_configure(tables, N_TABLES, map, 1024ull * 1024ull * 1024ull),
          "reconfigure with huge budget");
    check_u(ds4_qwen4_expert_cache_slot_count(), COUNT, "slots clamp to expert count");
    ds4_qwen4_expert_cache_shutdown();

    free(map);
    printf("%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
