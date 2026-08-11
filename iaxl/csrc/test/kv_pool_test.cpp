// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kv_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "[FAIL] %s (%s:%d)\n", (msg), __FILE__, __LINE__);                \
            ++g_failures;                                                                          \
        } else {                                                                                   \
            std::printf("[ok]   %s\n", (msg));                                                     \
        }                                                                                          \
    } while (0)

static char *dup_buf(const std::string &s) {
    char *p = static_cast<char *>(std::malloc(s.size()));
    std::memcpy(p, s.data(), s.size());
    return p;
}

static bool buf_eq(const char *p, size_t n, const std::string &s) {
    return p != nullptr && n == s.size() && std::memcmp(p, s.data(), n) == 0;
}

int main(int argc, char **argv) {
    const std::string dir = (argc > 1) ? argv[1] : "/tmp/kv_pool_test_dir";
    const std::string db_path = (argc > 2) ? argv[2] : "/tmp/kv_pool_test.db";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    {
        kv_pool::Storage storage(dir);
        const std::string label = "kv:abc123:0";
        const std::string payload = "hello-storage";
        CHECK(storage.save(label, payload.data(), payload.size()), "storage.save() ok");
        auto [buf, n] = storage.load(label);
        CHECK(buf_eq(buf, n, payload), "storage.load() returns saved bytes");
        std::free(buf);
    }

    {
        kv_pool::Mem mem(0);
        const std::vector<std::string> keys = {"kv:0:k", "kv:0:v", "kv:1:k"};
        const std::string d0 = "AAAA", d1 = "BBBBBB", d2 = "CC";

        std::vector<char *> ptrs = {dup_buf(d0), dup_buf(d1), dup_buf(d2)};
        std::vector<size_t> sizes = {d0.size(), d1.size(), d2.size()};
        mem.put(keys, std::move(ptrs), sizes);

        CHECK(mem.size() == 3, "size() == 3 after put");
        CHECK(mem.group_count() == 2, "group_count() == 2 (kv:0, kv:1)");
        CHECK(mem.current_bytes() == d0.size() + d1.size() + d2.size(),
              "current_bytes() sums buffers");

        auto got = mem.get(keys);
        CHECK(buf_eq(got[0].first, got[0].second, d0), "get() key0 matches");
        CHECK(buf_eq(got[1].first, got[1].second, d1), "get() key1 matches");
        CHECK(buf_eq(got[2].first, got[2].second, d2), "get() key2 matches");
        CHECK(mem.hits() == 3, "hits() == 3 after 3 gets");

        auto present = mem.has({"kv:0:k", "kv:9:x"});
        CHECK(present.size() == 2 && present[0] && !present[1], "has() reports presence");

        size_t freed = mem.remove("kv:0:k");
        CHECK(freed == d0.size() + d1.size(), "remove() frees whole group bytes");
        CHECK(mem.group_count() == 1, "group_count() == 1 after remove");
        CHECK(mem.size() == 1, "size() == 1 after remove");

        auto miss = mem.get({"kv:0:k"});
        CHECK(miss[0].first == nullptr, "get() after remove is a miss");
        CHECK(mem.misses() >= 1, "misses() incremented");
    }

    {
        kv_pool::Mem mem(0);
        std::vector<std::string> keys;
        std::vector<char *> ptrs;
        std::vector<size_t> sizes;
        for (int i = 0; i < 4; ++i) {
            keys.push_back("kv:" + std::to_string(i) + ":k");
            std::string d(100, 'x');
            ptrs.push_back(dup_buf(d));
            sizes.push_back(d.size());
        }
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.current_bytes() == 400, "current_bytes() == 400 before evict");

        size_t evicted = mem.evict_to_size(150);
        CHECK(evicted >= 250, "evict_to_size() freed oldest groups");
        CHECK(mem.current_bytes() <= 150, "current_bytes() <= target after evict");
        CHECK(mem.evictions() >= 1, "evictions() incremented");
    }

    {
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(0, &storage, &record);

        const std::vector<std::string> keys = {"kv:aa:k", "kv:bb:k"};
        const std::string da = "payload-A", db = "payload-B";
        std::vector<char *> ptrs = {dup_buf(da), dup_buf(db)};
        std::vector<size_t> sizes = {da.size(), db.size()};

        record.submit("kv", {"aa", "bb"});
        record.sync();
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.unpersisted_count() == 2, "unpersisted_count() == 2 after put");

        auto persisted = mem.persist_groups(10);
        CHECK(persisted.size() == 2, "persist_groups() persisted 2 groups");
        CHECK(mem.unpersisted_count() == 0, "unpersisted_count() == 0 after persist");

        size_t evicted = mem.force_evict_to_size(0);
        CHECK(evicted == da.size() + db.size(), "force_evict_to_size(0) cleared memory");
        CHECK(mem.group_count() == 0, "group_count() == 0 after full evict");

        auto reloaded = mem.get(keys, true);
        CHECK(buf_eq(reloaded[0].first, reloaded[0].second, da), "get() reloads key0 from storage");
        CHECK(buf_eq(reloaded[1].first, reloaded[1].second, db), "get() reloads key1 from storage");
        CHECK(mem.hits_in_storage() == 2, "hits_in_storage() == 2 after reload");
    }

    std::filesystem::remove_all(dir, ec);
    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    if (g_failures == 0) {
        std::printf("\nAll kv_pool checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d kv_pool check(s) failed.\n", g_failures);
    return 1;
}
