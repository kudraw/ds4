/*
 * Resident routed-expert weight cache for Qwen3.8-Flash-Next on discrete CUDA.
 * See ds4_qwen4_expert_cache.h for the design.  Host logic only; device
 * memory and kernel-side resolution live in the CUDA backend.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ds4_qwen4_expert_cache.h"
#include "ds4_gpu.h"

#define QEXPERT_TYPE_Q8_0 8u
#define QEXPERT_TYPE_MXFP4 39u
#define QEXPERT_ALIGN 32u

typedef struct {
    const void *map;      /* engine model map identity, matches weights() */
    uint64_t begin;       /* table start in the model file */
    uint64_t row;         /* one expert's bytes in the file */
    uint64_t stride;      /* one expert's bytes in the slab (aligned) */
} qexpert_table;

typedef struct {
    int32_t triple; /* physical layer owning the cached rows, -1 free */
    int32_t expert;
    uint64_t used;
} qexpert_slot;

static qexpert_table *g_qex_tables;
static size_t g_qex_n_tables;
/* Shared resident pool: ONE slab per table-kind (0=gate,1=up,2=down), reused
 * across every layer.  A slot row holds whichever (layer,expert) the `triple`
 * tag names; a later layer's miss overwrites it.  VRAM is therefore
 * 3 * slot_count * kind_stride, independent of the layer count - the whole
 * point of tagging slots by triple.  (Reserving a slab per routed table would
 * cost n_tables * slot_count and exhaust a discrete GPU's VRAM.) */
static ds4_gpu_tensor *g_qex_slab[3];
static char *g_qex_slab_base[3];
static uint64_t g_qex_kind_stride[3];
static int32_t *g_qex_map; /* [n_triples][count] table-local slot or -1 */
static qexpert_slot *g_qex_slots;
static uint32_t g_qex_slot_count;
static uint64_t g_qex_clock;
static uint32_t g_qex_count; /* experts per table */
static int g_qex_open_triple = -1;
static uint64_t g_qex_stat_hits, g_qex_stat_misses, g_qex_stat_steals, g_qex_stat_staged_bytes;
static int g_qex_stats_on;

static uint64_t qex_align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

static uint64_t tb_row_bytes(uint32_t triple) {
    return g_qex_tables[triple * 3].row;
}

static int32_t *qex_map_row(uint32_t triple) {
    return g_qex_map + (uint64_t)triple * g_qex_count;
}

void ds4_qwen4_expert_cache_shutdown(void) {
    ds4_qwen4_expert_cache_unstage();
    for (int j = 0; j < 3; j++) {
        if (g_qex_slab[j]) {
            ds4_gpu_qwen4_expert_slab_release(g_qex_slab[j]);
            g_qex_slab[j] = NULL;
        }
        g_qex_slab_base[j] = NULL;
        g_qex_kind_stride[j] = 0;
    }
    if (g_qex_tables) {
        free(g_qex_tables);
    }
    free(g_qex_map);
    free(g_qex_slots);
    g_qex_tables = NULL;
    g_qex_n_tables = 0;
    g_qex_map = NULL;
    g_qex_slots = NULL;
    g_qex_slot_count = 0;
    g_qex_count = 0;
    g_qex_open_triple = -1;
}

void ds4_qwen4_expert_cache_set_stats(int on) {
    g_qex_stats_on = on ? 1 : 0;
}

bool ds4_qwen4_expert_cache_configure(const ds4_qwen4_expert_table *tables, size_t n_tables,
                                      const void *model_map, uint64_t budget_bytes) {
    ds4_qwen4_expert_cache_shutdown();
    if (!tables || n_tables == 0 || n_tables % 3 != 0 || !model_map || budget_bytes == 0)
        return false;
    const uint32_t count = tables[0].count;
    if (count == 0)
        return false;
    /* Resident cost of one slot = the three kind rows (gate+up+down).  Each
     * kind slab is shared by all layers, so the layer count does NOT enter
     * per_slot.  kind_stride[j] is the max row stride over all layers for that
     * kind so a single shared stride indexes every layer's rows. */
    uint64_t kind_stride[3] = {0, 0, 0};
    for (size_t i = 0; i < n_tables; i++) {
        if (tables[i].count != count || tables[i].bytes == 0 ||
            tables[i].bytes % count != 0)
            return false;
        if (tables[i].type != QEXPERT_TYPE_Q8_0 && tables[i].type != QEXPERT_TYPE_MXFP4)
            return false;
        const uint64_t stride_i = qex_align_up(tables[i].bytes / count, QEXPERT_ALIGN);
        const size_t j = i % 3;
        if (stride_i > kind_stride[j])
            kind_stride[j] = stride_i;
    }
    uint64_t per_slot = kind_stride[0] + kind_stride[1] + kind_stride[2];
    if (per_slot == 0 || budget_bytes < per_slot)
        return false; /* not even one expert triple fits */
    uint64_t slots = budget_bytes / per_slot;
    /* Slot indices travel in the expert-id field but are NOT raw ids: ds4.c
     * re-encodes each as a sentinel (count + slot) before the kernels see it
     * (see ds4.c qwen4_graph_moe_stage_experts and moe_mv in
     * ds4_qwen4_cuda.cuh).  The id domain therefore has two disjoint bands:
     * raw expert ids [0, count) (disk rows, cache window closed) and resident
     * sentinels [count, count + slots) (slab rows, window open).  Because the
     * sentinel band starts at count it can never alias a raw id, so slot_count
     * is free to exceed the per-layer expert count - that is the whole point:
     * a decode batch touches far more distinct experts across the layers it
     * spans than any single layer routes, and a global LRU over one shared
     * pool exploits that.  The sole remaining constraint is that a sentinel
     * fits in the int32 id field the kernels read. */
    if (slots + count > (uint64_t)INT32_MAX)
        slots = (uint64_t)INT32_MAX - count;
    if (slots > UINT32_MAX)
        return false;

    g_qex_tables = calloc(n_tables, sizeof(*g_qex_tables));
    g_qex_map = calloc((n_tables / 3) * (size_t)count, sizeof(*g_qex_map));
    g_qex_slots = calloc((size_t)slots, sizeof(*g_qex_slots));
    if (!g_qex_tables || !g_qex_map || !g_qex_slots) {
        free(g_qex_tables);
        free(g_qex_map);
        free(g_qex_slots);
        g_qex_tables = NULL;
        g_qex_map = NULL;
        g_qex_slots = NULL;
        return false;
    }
    g_qex_n_tables = n_tables;
    g_qex_count = count;
    g_qex_slot_count = (uint32_t)slots;
    for (size_t i = 0; i < n_tables; i++) {
        g_qex_tables[i].map = model_map;
        g_qex_tables[i].begin = tables[i].file_offset;
        g_qex_tables[i].row = tables[i].bytes / count;
        g_qex_tables[i].stride = qex_align_up(g_qex_tables[i].row, QEXPERT_ALIGN);
    }
    for (int j = 0; j < 3; j++)
        g_qex_kind_stride[j] = kind_stride[j];
    for (size_t t = 0; t < n_tables / 3; t++) {
        int32_t *row = qex_map_row((uint32_t)t);
        for (uint32_t e = 0; e < count; e++)
            row[e] = -1;
    }
    /* calloc would tag every slot with layer 0 (triple 0 is a valid tag);
     * free slots must read as triple < 0 or the steal scan never sees them. */
    for (uint64_t i = 0; i < slots; i++) {
        g_qex_slots[i].triple = -1;
        g_qex_slots[i].expert = -1;
    }
    /* Three shared slabs (gate/up/down), each slot_count rows, hold the whole
     * resident pool for every layer at once via triple-tagged slot reuse. */
    g_qex_stats_on = getenv("DS4_QWEN4_EXPERT_CACHE_STATS") != NULL;
    for (int j = 0; j < 3; j++) {
        g_qex_slab[j] = ds4_gpu_qwen4_expert_slab_reserve(g_qex_slot_count * g_qex_kind_stride[j]);
        if (!g_qex_slab[j]) {
            ds4_qwen4_expert_cache_shutdown();
            return false;
        }
        g_qex_slab_base[j] = (char *)ds4_gpu_tensor_contents(g_qex_slab[j]);
    }
    return true;
}

bool ds4_qwen4_expert_cache_enabled(void) {
    return g_qex_tables != NULL && g_qex_slot_count > 0;
}

uint32_t ds4_qwen4_expert_cache_slot_count(void) {
    return g_qex_slot_count;
}

bool ds4_qwen4_expert_cache_lookup(uint32_t phys_layer, uint32_t expert, int32_t *slot) {
    if (!ds4_qwen4_expert_cache_enabled() || phys_layer >= g_qex_n_tables / 3 ||
        expert >= g_qex_count)
        return false;
    *slot = qex_map_row(phys_layer)[expert];
    return true;
}

/* Pick a victim slot: globally least recently used, excepting slots this
 * stage call already published in `slots` (kernels resolve those indices to
 * bytes and must not see them change).  Tables keep independent slabs, so
 * evicting a slot tagged with another triple only discards that layer's
 * cached expert; it does not free bytes "for" the current layer.  A plain
 * global LRU is therefore correct and keeps interleaved layers from
 * destroying each other's warm rows, which a current-triple-exempt policy
 * would: with 48 layers touching the pool per token, every layer's rows get
 * stolen the moment the next layer stages.  The scan is O(slots * ids) and
 * slot_count is capped at the expert count. */
static int32_t qex_victim(uint32_t triple, const int32_t *slots) {
    (void)triple;
    int32_t best = -1;
    uint64_t best_used = UINT64_MAX;
    for (uint32_t i = 0; i < g_qex_slot_count; i++) {
        const qexpert_slot *s = &g_qex_slots[i];
        if (s->triple == (int32_t)triple) {
            int pinned = 0;
            for (uint32_t e = 0; e < g_qex_count; e++) {
                if (slots[e] == (int32_t)i) {
                    pinned = 1;
                    break;
                }
            }
            if (pinned)
                continue;
        }
        if (s->used < best_used) {
            best_used = s->used;
            best = (int32_t)i;
        }
    }
    /* -1 only when every slot is pinned by this call: more distinct ids
     * than the slab holds.  Fail the stage (the window stays closed and
     * kernels use raw ids) rather than evict a published row. */
    return best;
}

static bool qex_load(uint32_t triple, uint32_t expert, int32_t slot) {
    const qexpert_slot *old = &g_qex_slots[slot];
    if (old->triple >= 0 && old->expert >= 0)
        qex_map_row((uint32_t)old->triple)[old->expert] = -1;
    for (uint32_t j = 0; j < 3; j++) {
        const qexpert_table *tb = &g_qex_tables[triple * 3 + j];
        const char *src = (const char *)tb->map + tb->begin + (uint64_t)expert * tb->row;
        if (ds4_gpu_tensor_write(g_qex_slab[j], (uint64_t)slot * g_qex_kind_stride[j], src,
                                 tb->row) == 0)
            return false;
    }
    g_qex_slots[slot].triple = (int32_t)triple;
    g_qex_slots[slot].expert = (int32_t)expert;
    g_qex_slots[slot].used = ++g_qex_clock;
    g_qex_stat_staged_bytes += 3 * tb_row_bytes(triple);
    qex_map_row(triple)[expert] = slot;
    return true;
}

bool ds4_qwen4_expert_cache_stage(uint32_t phys_layer, const int32_t *ids, uint32_t n_ids,
                                  int32_t *slots) {
    if (!ds4_qwen4_expert_cache_enabled() || phys_layer >= g_qex_n_tables / 3 ||
        (!ids && n_ids) || !slots)
        return false;
    int32_t *row = qex_map_row(phys_layer);
    for (uint32_t e = 0; e < g_qex_count; e++)
        slots[e] = -1;
    for (uint32_t i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e < 0 || (uint32_t)e >= g_qex_count)
            continue; /* caller clamps; routing noise must not corrupt maps */
        slots[e] = row[e];
        if (slots[e] >= 0) {
            g_qex_slots[slots[e]].used = ++g_qex_clock;
            g_qex_stat_hits++;
            continue;
        }
        g_qex_stat_misses++;
        const int32_t v = qex_victim(phys_layer, slots);
        if (v < 0)
            return false; /* window stays closed; kernels use raw ids */
        if (g_qex_slots[v].triple >= 0)
            g_qex_stat_steals++;
        if (!qex_load(phys_layer, (uint32_t)e, v)) {
            ds4_qwen4_expert_cache_shutdown();
            return false;
        }
        slots[e] = row[e];
    }
    uint64_t begins[3], ends[3], rows[3], strides[3];
    char *slabs[3];
    const void *map = g_qex_tables[phys_layer * 3].map;
    for (uint32_t j = 0; j < 3; j++) {
        const qexpert_table *tb = &g_qex_tables[phys_layer * 3 + j];
        begins[j] = tb->begin;
        ends[j] = tb->begin + (uint64_t)g_qex_count * tb->row;
        rows[j] = tb->row;
        strides[j] = g_qex_kind_stride[j];
        slabs[j] = g_qex_slab_base[j];
    }
    if (ds4_gpu_qwen4_expert_set_window(map, begins, ends, rows, strides, slabs) != 0) {
        /* Loads above already filled the host maps; without the published
         * window those slot indices mean nothing to the kernels, so drop
         * the whole cache rather than let lookup() report a residency the
         * device cannot resolve.  Kernels stay on raw ids from here on. */
        ds4_qwen4_expert_cache_shutdown();
        return false;
    }
    g_qex_open_triple = (int)phys_layer;
    return true;
}

void ds4_qwen4_expert_cache_unstage(void) {
    if (g_qex_open_triple < 0)
        return;
    g_qex_open_triple = -1;
    ds4_gpu_qwen4_expert_set_window(NULL, NULL, NULL, NULL, NULL, NULL);
}

/* Diagnostic readback (engine cache probe): copy one staged expert row back
 * from the slab into dst.  Returns false when the expert is not resident
 * (or the layer/table index is out of range).  bytes is clamped to the
 * table row size. */
bool ds4_qwen4_expert_cache_slab_read(uint32_t phys_layer, uint32_t expert,
                                      uint32_t table_idx, void *dst, uint64_t bytes) {
    if (!ds4_qwen4_expert_cache_enabled() || phys_layer >= g_qex_n_tables / 3 ||
        expert >= g_qex_count || table_idx >= 3 || !dst)
        return false;
    const int32_t slot = qex_map_row(phys_layer)[expert];
    if (slot < 0)
        return false;
    const qexpert_table *tb = &g_qex_tables[phys_layer * 3 + table_idx];
    if (bytes > tb->row)
        bytes = tb->row;
    return ds4_gpu_tensor_read(g_qex_slab[table_idx],
                               (uint64_t)slot * g_qex_kind_stride[table_idx], dst,
                               bytes) != 0;
}

void ds4_qwen4_expert_cache_log_stats_final(void) {
    ds4_qwen4_expert_cache_log_stats("final");
}

void ds4_qwen4_expert_cache_log_stats(const char *tag) {
    if (!g_qex_stats_on || !ds4_qwen4_expert_cache_enabled())
        return;
    const uint64_t total = g_qex_stat_hits + g_qex_stat_misses;
    fprintf(stderr,
            "[qwen4-expert-cache %s] slot_pool=%u hits=%llu misses=%llu (%.1f%%) "
            "steals=%llu staged=%.2f GiB\n",
            tag ? tag : "", (unsigned)g_qex_slot_count,
            (unsigned long long)g_qex_stat_hits, (unsigned long long)g_qex_stat_misses,
            total ? 100.0 * (double)g_qex_stat_hits / (double)total : 0.0,
            (unsigned long long)g_qex_stat_steals,
            (double)g_qex_stat_staged_bytes / (1024.0 * 1024.0 * 1024.0));
}
