/*
 * Resident routed-expert weight cache for Qwen3.8-Flash-Next on discrete CUDA.
 *
 * The routed expert tables (gate/up/down per layer, one row per expert)
 * dominate the model, but any one forward touches only a few routed experts
 * per layer.  On a discrete card every weight read is a host read staged to
 * the device (pinned ranges, or the host-resident q4x arena), which caps
 * decode bandwidth.  This cache keeps a capacity-bounded set of expert rows
 * resident in device memory for the current layer.  Within one layer's
 * staging window the CUDA weight resolver maps a routed expert's file offset
 * to its device slab row; the engine rewrites the device routing ids to slab
 * slot indices, so the expert kernels only ever resolve resident rows.
 * Dense and shared-expert weights sit outside the three registered table
 * ranges and resolve as usual.
 *
 * Stage takes the expert ids as a host array and writes the resolved device
 * slots back as a host array; converting those between device and host is
 * the engine's job.
 *
 * Slots are shared across layers with a per-slot layer tag: a slot keeps its
 * layer's expert until a later layer's miss steals it, so repeat forwards
 * over the same routing (the common case for a decoding batch) reload
 * nothing once warm.  Loading reads the row bytes straight from the mapped
 * model file (the same host base the resolver uses) and copies them into the
 * slab; because cached rows never pass through the resolver, they are not
 * separately pinned or copied into the q4x arena.
 *
 * Host logic only: this file performs no CUDA calls itself.  Device memory
 * comes from ds4_gpu_qwen4_expert_slab_reserve() and window activation goes
 * through ds4_gpu_qwen4_expert_set_window(), both CUDA backend.  Where the
 * backend declines (integrated GPUs, SSD mode, allocation failure) the cache
 * stays disabled and the regular paths run.
 */
#ifndef DS4_QWEN4_EXPERT_CACHE_H
#define DS4_QWEN4_EXPERT_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* One routed expert table: `bytes` of `count` equal consecutive expert rows
 * at `file_offset` in the mapped model file.  The three tables of one
 * physical layer must be passed adjacent, gate, up, down. */
typedef struct {
    uint64_t file_offset; /* absolute model-file byte offset of row 0 */
    uint64_t bytes;       /* count * file row bytes */
    uint32_t count;       /* experts per table */
    uint32_t type;        /* GGUF type id: q8_0 (8) or mxfp4 (39) */
} ds4_qwen4_expert_table;

/* Lay out the cache for `n_tables` tables and a total device budget.
 * `model_map` is the engine's mapped model base (host pointer).  Returns
 * false when the budget cannot hold a single expert triple or the tables are
 * not cacheable (mixed expert row counts, unsupported types).  Valid until
 * shutdown; call before any staging. */
bool ds4_qwen4_expert_cache_configure(const ds4_qwen4_expert_table *tables, size_t n_tables,
                                      const void *model_map, uint64_t budget_bytes);
void ds4_qwen4_expert_cache_shutdown(void);
bool ds4_qwen4_expert_cache_enabled(void);

/* Device slab slots per layer table after a successful configure. */
uint32_t ds4_qwen4_expert_cache_slot_count(void);

/* Hit/miss counters to stderr when on (configure also honours
 * DS4_QWEN4_EXPERT_CACHE_STATS). */
void ds4_qwen4_expert_cache_set_stats(int on);
void ds4_qwen4_expert_cache_log_stats(const char *tag);
void ds4_qwen4_expert_cache_log_stats_final(void);

/* Current slot holding one expert's rows, or -1 when not resident.  Fails
 * only for out-of-range ids or when the cache is off. */
bool ds4_qwen4_expert_cache_lookup(uint32_t phys_layer, uint32_t expert, int32_t *slot);

/* Stage a layer for kernel recording: load every expert in `ids` (LRU,
 * layer-tagged), publish the CUDA resolution window, and write each expert's
 * device slot into `slots` (index by expert id, entries untouched by `ids`
 * set to -1).  Serializes with device work (it syncs to copy rows). */
bool ds4_qwen4_expert_cache_stage(uint32_t phys_layer, const int32_t *ids, uint32_t n_ids, int32_t *slots);

/* Close the window opened by the last stage.  Idempotent. */
void ds4_qwen4_expert_cache_unstage(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_QWEN4_EXPERT_CACHE_H */
