// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// QAT, IAA and CPU workers share one task pool. Each worker claims another item when a request
// completes, so the faster backend naturally processes more of the batch. QAT and IAA workers keep
// multiple asynchronous requests in flight, while CPU workers run one synchronous raw-DEFLATE
// request each. Up to IAXL_OMP_THREAD_NUM threads (see parallel.h) drive the workers, each polling
// the slots of its workers without blocking, so a single thread can drive them all.
//
// An item is one segment of a tensor: the whole tensor, or IAXL_ZIP_SEGMENT_SIZE_KB of it so that a
// large tensor keeps several requests in flight. Each tensor still becomes exactly one block: a
// header, a segment table and the segment payloads.
//
// QAT and CPU streams are mutually compatible, IAA streams are compatible with neither, so each
// segment records whether IAA produced it and decompression only hands it to a backend that can
// decode it.

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "env.h"
#include "iaxl_common.h"
#include "parallel.h"
#include "cpu_zip.h"
#include "iaa_zip.h"
#include "qat_zip.h"
#include "data_shuffle.h"
#include "lossy.h"
#include "kv_zip.h"

namespace kv_zip {

// A block holds one tensor: a BlockHeader, `segments` SegmentEntry records, then the payloads.
struct BlockHeader {
    uint32_t magic;        // kBlockMagic
    uint16_t version;      // kBlockVersion
    uint16_t flags;        // kBlockShuffled
    uint32_t orig_size;    // tensor bytes
    uint32_t segment_size; // tensor bytes per segment, the last one also takes the remainder
    uint32_t segments;     // SegmentEntry records that follow
};

struct SegmentEntry {
    uint32_t offset; // payload start, from the start of the block
    uint32_t length; // payload bytes
    uint32_t flags;  // kSegmentStored, kSegmentIaa
};

static_assert(sizeof(BlockHeader) == 20 && sizeof(SegmentEntry) == 12,
              "the block layout is persisted and must not change");

static constexpr uint32_t kBlockMagic = 0x5A584149; // "IAXZ" in memory
static constexpr uint16_t kBlockVersion = 1;
// Every segment was byte-shuffled on its own before compression.
static constexpr uint16_t kBlockShuffled = 1u << 0;
// The payload is the segment itself: compression was off or did not make it smaller.
static constexpr uint32_t kSegmentStored = 1u << 0;
// IAA produced the stream, so only IAA can decode it.
static constexpr uint32_t kSegmentIaa = 1u << 1;

// One segment of one block in a batch.
struct SegmentRef {
    uint32_t block;
    uint32_t index;
};

static uint32_t segment_count(uint32_t orig_size, uint32_t segment_size) {
    return std::max<uint32_t>(1, orig_size / segment_size);
}

// IAXL_ZIP_SEGMENT_SIZE_KB in bytes; 0, or a size not below the tensor's, keeps one segment.
static uint32_t segment_size_for(uint32_t orig_size) {
    const uint64_t size = uint64_t(envs.IAXL_ZIP_SEGMENT_SIZE_KB) * 1024;
    if (size == 0 || size >= orig_size)
        return std::max<uint32_t>(orig_size, 1);
    return static_cast<uint32_t>(size);
}

static BlockHeader make_header(size_t orig_size, uint16_t flags) {
    const auto size = static_cast<uint32_t>(orig_size);
    const uint32_t segment_size = segment_size_for(size);
    const uint32_t segments = segment_count(size, segment_size);
    return {kBlockMagic, kBlockVersion, flags, size, segment_size, segments};
}

// Tensor bytes of segment k, which starts at k * segment_size.
static uint32_t segment_length(const BlockHeader &h, uint32_t k) {
    return k + 1 < h.segments ? h.segment_size : h.orig_size - k * h.segment_size;
}

static size_t payload_start(uint32_t segments) {
    return sizeof(BlockHeader) + segments * sizeof(SegmentEntry);
}

static const BlockHeader &block_header(const char *block) {
    return *reinterpret_cast<const BlockHeader *>(block);
}

static const SegmentEntry *segment_table(const char *block) {
    return reinterpret_cast<const SegmentEntry *>(block + sizeof(BlockHeader));
}

static size_t tensor_bytes(const torch::Tensor &t) { return t.numel() * t.element_size(); }

enum class ZipBackend { QAT, IAA, CPU };

// Backend entry points, indexed by ZipBackend.
struct ZipOps {
    int (*compress)(int slot, void *src, int len);
    int (*decompress)(int slot, void *src, int len);
    int (*wait)(int slot, void **dest, int *len, int non_block);
    int (*src_cap)(void);
    int (*queue_depth)(void);
};

static const ZipOps kZipOps[] = {
    {qat_zip_compress, qat_zip_decompress, qat_zip_wait, qat_zip_src_cap, qat_zip_queue_depth},
    {iaa_zip_compress, iaa_zip_decompress, iaa_zip_wait, iaa_zip_src_cap, iaa_zip_queue_depth},
    {cpu_zip_compress, cpu_zip_decompress, cpu_zip_wait, cpu_zip_src_cap, cpu_zip_queue_depth},
};

static const ZipOps &ops(ZipBackend backend) { return kZipOps[static_cast<int>(backend)]; }

// Returned by get_next once a backend has nothing left to claim.
static constexpr size_t kNoTask = static_cast<size_t>(-1);

static void ensure_zip_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        if (envs.IAXL_QAT_ZIP_ENABLE)
            IAXL_CHECK(qat_zip_init() == 0, "kv_zip: qat_zip_init failed");
        if (envs.IAXL_IAA_ZIP_ENABLE)
            IAXL_CHECK(iaa_zip_init() == 0, "kv_zip: iaa_zip_init failed");
        if (envs.IAXL_CPU_ZIP_ENABLE)
            IAXL_CHECK(cpu_zip_init() == 0, "kv_zip: cpu_zip_init failed");
    });
}

template <class Next, class Submit, class Complete>
static void zip_pipeline(Next &&get_next, Submit &&submit, Complete &&complete) {
    ensure_zip_init();
    const int qat_workers = envs.IAXL_QAT_ZIP_ENABLE ? envs.IAXL_QAT_INSTANCE_NUM : 0;
    const int iaa_workers = envs.IAXL_IAA_ZIP_ENABLE ? envs.IAXL_IAA_INSTANCE_NUM : 0;
    const int cpu_workers = envs.IAXL_CPU_ZIP_ENABLE ? cpu_zip_num_slots() : 0;
    const int worker_count = qat_workers + iaa_workers + cpu_workers;
    IAXL_CHECK(qat_workers == 0 || qat_workers <= qat_zip_num_slots() / qat_zip_queue_depth(),
               "kv_zip: IAXL_QAT_INSTANCE_NUM exceeds available QAT instances");
    IAXL_CHECK(iaa_workers == 0 || iaa_workers <= iaa_zip_num_slots() / iaa_zip_queue_depth(),
               "kv_zip: IAXL_IAA_INSTANCE_NUM exceeds available IAA instances");
    const int threads = std::min(envs.IAXL_OMP_THREAD_NUM, worker_count);

    // Thread `tid` owns workers tid, tid + threads, ..., so each backend instance stays on one
    // thread while the backends are spread across all of them.
    auto drive = [&](int tid) {
        struct Lane {
            ZipBackend backend;
            int slot;
            size_t item;
        };
        std::vector<Lane> lanes;
        for (int w = tid; w < worker_count; w += threads) {
            ZipBackend backend = ZipBackend::CPU;
            int first = qat_workers + iaa_workers;
            if (w < qat_workers) {
                backend = ZipBackend::QAT;
                first = 0;
            } else if (w < qat_workers + iaa_workers) {
                backend = ZipBackend::IAA;
                first = qat_workers;
            }
            const int depth = ops(backend).queue_depth();
            for (int k = 0; k < depth; k++) {
                const size_t i = get_next(backend);
                if (i == kNoTask)
                    break;
                const int slot = (w - first) * depth + k;
                submit(backend, slot, i);
                lanes.push_back({backend, slot, i});
            }
        }

        while (!lanes.empty()) {
            for (size_t s = 0; s < lanes.size();) {
                Lane &lane = lanes[s];
                void *out;
                int out_len;
                const int status =
                    ops(lane.backend).wait(lane.slot, &out, &out_len, /*non_block=*/1);
                if (status == IAXL_ZIP_PENDING) {
                    s++;
                    continue;
                }
                IAXL_CHECK(status == 0, "kv_zip: zip wait failed");
                complete(lane.backend, lane.item, out, out_len);

                lane.item = get_next(lane.backend);
                if (lane.item == kNoTask) {
                    lane = lanes.back();
                    lanes.pop_back();
                    continue;
                }
                submit(lane.backend, lane.slot, lane.item);
                s++;
            }
        }
    };

    parallel::run_threads(threads, drive);
}

// Compression state of one tensor. Segments finishing while others are still in flight park their
// streams in `staging`, and the last one to finish assembles the block.
struct ZipBlock {
    char *data = nullptr; // tensor bytes, prepared in place one segment at a time
    int element_size = 0;
    BlockHeader header{};
    uint32_t first = 0; // position of segment 0 in the batch's segment list
    std::atomic<uint32_t> pending{0};
    std::atomic<char *> staging{nullptr};
};

void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress) {
    const size_t n = tensors.size();
    for (size_t i = 0; i < n; i++) {
        const auto &t = tensors[i];
        IAXL_CHECK(t.is_contiguous() && t.device().type() == c10::DeviceType::CPU,
                   "kv_zip: tensor must be a contiguous CPU tensor");
        orig_sizes[i] = tensor_bytes(t);
        IAXL_CHECK(orig_sizes[i] <= static_cast<size_t>(INT_MAX),
                   "kv_zip: tensor byte size exceeds zip integer length range");
    }
    if (n > 0) {
        static std::once_flag logged;
        std::call_once(logged, [&] {
            const BlockHeader h = make_header(orig_sizes[0], 0);
            printf("[kv_zip] block_size=%u B segment_size=%u B segments=%u\n", h.orig_size,
                   h.segment_size, h.segments);
            fflush(stdout);
        });
    }

    if (!compress || !envs.IAXL_KV_COMPRESSION) {
        parallel::parallel_for(n, [&](size_t i) {
            const BlockHeader h = make_header(orig_sizes[i], 0);
            const size_t start = payload_start(h.segments);
            char *buf = static_cast<char *>(malloc(start + h.orig_size));
            IAXL_CHECK(buf != nullptr, "kv_zip: raw cache buffer allocation failed");
            memcpy(buf, &h, sizeof(h));
            auto *table = reinterpret_cast<SegmentEntry *>(buf + sizeof(h));
            for (uint32_t k = 0; k < h.segments; k++)
                table[k] = {static_cast<uint32_t>(start + size_t(k) * h.segment_size),
                            segment_length(h, k), kSegmentStored};
            memcpy(buf + start, tensors[i].data_ptr(), h.orig_size);
            out_bufs[i] = buf;
            out_sizes[i] = start + h.orig_size;
        });
        return;
    }

    std::vector<ZipBlock> blocks(n);
    std::vector<SegmentRef> segs;
    segs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const auto &t = tensors[i];
        ZipBlock &b = blocks[i];
        b.data = static_cast<char *>(t.data_ptr());
        b.element_size = static_cast<int>(t.element_size());
        const bool shuffle = data_shuffle_enabled() && t.dtype() == torch::kBFloat16;
        b.header = make_header(orig_sizes[i], shuffle ? kBlockShuffled : 0);
        b.first = static_cast<uint32_t>(segs.size());
        b.pending.store(b.header.segments, std::memory_order_relaxed);
        for (uint32_t k = 0; k < b.header.segments; k++)
            segs.push_back({static_cast<uint32_t>(i), k});
    }
    std::vector<SegmentEntry> entries(segs.size());

    auto staging_of = [](ZipBlock &b) {
        char *buf = b.staging.load(std::memory_order_acquire);
        if (buf)
            return buf;
        char *fresh = static_cast<char *>(malloc(b.header.orig_size));
        IAXL_CHECK(fresh != nullptr, "kv_zip: staging buffer allocation failed");
        if (b.staging.compare_exchange_strong(buf, fresh, std::memory_order_acq_rel))
            return fresh;
        free(fresh);
        return buf;
    };

    // `stream` is set when segment `last` skipped the staging buffer.
    auto assemble = [&](uint32_t i, uint32_t last, const char *stream) {
        ZipBlock &b = blocks[i];
        const BlockHeader &h = b.header;
        SegmentEntry *table = &entries[b.first];
        char *parked = b.staging.load(std::memory_order_acquire);
        size_t size = payload_start(h.segments);
        for (uint32_t k = 0; k < h.segments; k++)
            size += table[k].length;
        char *buf = static_cast<char *>(malloc(size));
        IAXL_CHECK(buf != nullptr, "kv_zip: cache buffer allocation failed");
        size_t offset = payload_start(h.segments);
        for (uint32_t k = 0; k < h.segments; k++) {
            const size_t at = size_t(k) * h.segment_size;
            const char *src = (table[k].flags & kSegmentStored) ? b.data + at
                              : (k == last && stream)            ? stream
                                                                 : parked + at;
            memcpy(buf + offset, src, table[k].length);
            table[k].offset = static_cast<uint32_t>(offset);
            offset += table[k].length;
        }
        memcpy(buf, &h, sizeof(h));
        memcpy(buf + sizeof(h), table, h.segments * sizeof(SegmentEntry));
        free(parked);
        out_bufs[i] = buf;
        out_sizes[i] = size;
    };

    std::atomic<size_t> next{0};
    zip_pipeline(
        [&](ZipBackend) {
            const size_t s = next.fetch_add(1, std::memory_order_relaxed);
            return s < segs.size() ? s : kNoTask;
        },
        [&](ZipBackend backend, int slot, size_t s) {
            const SegmentRef r = segs[s];
            ZipBlock &b = blocks[r.block];
            char *src = b.data + size_t(r.index) * b.header.segment_size;
            const uint32_t len = segment_length(b.header, r.index);
            IAXL_CHECK(len <= static_cast<uint32_t>(ops(backend).src_cap()),
                       "kv_zip: segment exceeds IAXL_ZIP_SRC_CAP; raise it or set "
                       "IAXL_ZIP_SEGMENT_SIZE_KB");
            lossy_trunc(src, len, b.element_size);
            data_shuffle(src, len, true, (b.header.flags & kBlockShuffled) != 0);
            const int status = ops(backend).compress(slot, src, static_cast<int>(len));
            IAXL_CHECK(status == 0, "kv_zip: zip compress failed");
        },
        [&](ZipBackend backend, size_t s, void *out, int out_len) {
            const SegmentRef r = segs[s];
            ZipBlock &b = blocks[r.block];
            const uint32_t len = segment_length(b.header, r.index);
            const char *stream = static_cast<const char *>(out);
            if (static_cast<uint32_t>(out_len) < len) {
                entries[s] = {0, static_cast<uint32_t>(out_len),
                              backend == ZipBackend::IAA ? kSegmentIaa : 0u};
            } else {
                // Not compressible: keep the prepared bytes, which are still in the tensor.
                entries[s] = {0, len, kSegmentStored};
                stream = nullptr;
            }
            if (b.pending.load(std::memory_order_acquire) > 1) {
                if (stream)
                    memcpy(staging_of(b) + size_t(r.index) * b.header.segment_size, stream,
                           out_len);
                if (b.pending.fetch_sub(1, std::memory_order_acq_rel) > 1)
                    return;
                stream = nullptr;
            }
            assemble(r.block, r.index, stream);
        });
}

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<torch::Tensor> &tensors) {
    const size_t n = tensors.size();
    IAXL_CHECK(data_ptrs.size() == n, "kv_zip: decompression inputs must have matching lengths");

    // IAA-produced segments must go back to IAA, the other compressed ones to QAT/CPU.
    std::vector<SegmentRef> stored, iaa, others;
    for (size_t i = 0; i < n; i++) {
        const BlockHeader &h = block_header(data_ptrs[i]);
        IAXL_CHECK(h.magic == kBlockMagic && h.version == kBlockVersion,
                   "kv_zip: unknown block format, e.g. a KV cache persisted by an older iaxl");
        IAXL_CHECK(h.orig_size == tensor_bytes(tensors[i]),
                   "kv_zip: block size does not match tensor byte size");
        IAXL_CHECK(h.segment_size > 0 && h.segments == segment_count(h.orig_size, h.segment_size),
                   "kv_zip: invalid segment table");
        const SegmentEntry *table = segment_table(data_ptrs[i]);
        for (uint32_t k = 0; k < h.segments; k++) {
            const SegmentRef r{static_cast<uint32_t>(i), k};
            if (table[k].flags & kSegmentStored)
                stored.push_back(r);
            else if (table[k].flags & kSegmentIaa)
                iaa.push_back(r);
            else
                others.push_back(r);
        }
    }

    auto place = [&](SegmentRef r, const void *src, size_t len) {
        const BlockHeader &h = block_header(data_ptrs[r.block]);
        IAXL_CHECK(len == segment_length(h, r.index),
                   "kv_zip: decompressed size does not match segment size");
        char *dst = static_cast<char *>(tensors[r.block].data_ptr()) +
                    size_t(r.index) * h.segment_size;
        memcpy(dst, src, len);
        data_shuffle(dst, len, true, (h.flags & kBlockShuffled) != 0);
    };

    parallel::parallel_for(stored.size(), [&](size_t j) {
        const SegmentRef r = stored[j];
        const SegmentEntry &e = segment_table(data_ptrs[r.block])[r.index];
        place(r, data_ptrs[r.block] + e.offset, e.length);
    });

    if (iaa.empty() && others.empty())
        return;
    IAXL_CHECK(iaa.empty() || envs.IAXL_IAA_ZIP_ENABLE,
               "kv_zip: batch holds IAA-compressed segments but the IAA backend is disabled");
    IAXL_CHECK(others.empty() || envs.IAXL_QAT_ZIP_ENABLE || envs.IAXL_CPU_ZIP_ENABLE,
               "kv_zip: batch holds QAT/CPU-compressed segments but both backends are disabled");

    // IAA segments first, then the others, each range with its own cursor.
    const size_t iaa_end = iaa.size();
    std::vector<SegmentRef> segs = std::move(iaa);
    segs.insert(segs.end(), others.begin(), others.end());
    std::atomic<size_t> iaa_next{0}, other_next{iaa_end};
    zip_pipeline(
        [&](ZipBackend backend) {
            if (backend == ZipBackend::IAA) {
                const size_t s = iaa_next.fetch_add(1, std::memory_order_relaxed);
                return s < iaa_end ? s : kNoTask;
            }
            const size_t s = other_next.fetch_add(1, std::memory_order_relaxed);
            return s < segs.size() ? s : kNoTask;
        },
        [&](ZipBackend backend, int slot, size_t s) {
            const SegmentRef r = segs[s];
            const SegmentEntry &e = segment_table(data_ptrs[r.block])[r.index];
            char *payload = const_cast<char *>(data_ptrs[r.block] + e.offset);
            const int status = ops(backend).decompress(slot, payload, static_cast<int>(e.length));
            IAXL_CHECK(status == 0, "kv_zip: zip decompress failed");
        },
        [&](ZipBackend, size_t s, void *out, int out_len) {
            place(segs[s], out, static_cast<size_t>(out_len));
        });
}

} // namespace kv_zip
