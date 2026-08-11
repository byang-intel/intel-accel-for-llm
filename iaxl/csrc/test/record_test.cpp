// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kv_pool.h"
#include "task_queue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

static bool all_true(const std::vector<bool> &v) {
    for (bool b : v)
        if (!b)
            return false;
    return true;
}

static bool none_true(const std::vector<bool> &v) {
    for (bool b : v)
        if (b)
            return false;
    return true;
}

int main(int argc, char **argv) {
    const std::string db_path = (argc > 1) ? argv[1] : "/tmp/record_test.db";

    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    const std::string label = "kv";
    const std::vector<std::string> chunk_ids = {"0", "1", "2"};
    const std::vector<std::string> group_keys = {"kv:0", "kv:1", "kv:2"};

    {
        TaskQueue queue("TQ-TEST");
        queue.init();

        std::promise<void> release_first;
        std::shared_future<void> first_gate(release_first.get_future());
        std::mutex order_mutex;
        std::vector<int> order;

        auto first = queue.submit([&]() { first_gate.wait(); });
        auto low = queue.submit(
            [&]() {
                std::lock_guard<std::mutex> lock(order_mutex);
                order.push_back(1);
            },
            TaskQueue::PRIORITY_LOW);
        auto high = queue.submit(
            [&]() {
                std::lock_guard<std::mutex> lock(order_mutex);
                order.push_back(0);
            },
            TaskQueue::PRIORITY_HIGH);
        release_first.set_value();
        first.get();
        high.get();
        low.get();
        CHECK(order == std::vector<int>({0, 1}), "TaskQueue runs HIGH before LOW");

        auto failed = queue.submit([]() { throw std::runtime_error("expected"); });
        bool exception_propagated = false;
        try {
            failed.get();
        } catch (const std::runtime_error &) {
            exception_propagated = true;
        }
        CHECK(exception_propagated, "TaskQueue future propagates task exception");

        queue.shutdown();
        bool submit_rejected = false;
        try {
            queue.submit([]() {});
        } catch (const std::runtime_error &) {
            submit_rejected = true;
        }
        CHECK(submit_rejected, "TaskQueue rejects submit after shutdown");
    }

    {
        kv_pool::Record rec(db_path, true);

        CHECK(none_true(rec.has(label, chunk_ids)), "has() is false before submit");

        rec.submit(label, chunk_ids);
        rec.sync();
        CHECK(rec.pending_count() == 0, "pending_count() is 0 after sync");
        CHECK(all_true(rec.has(label, chunk_ids)), "has() is true after submit");
        CHECK(none_true(rec.is_persisted(group_keys)), "is_persisted() is false after submit");

        rec.mark_persisted(group_keys);
        CHECK(all_true(rec.is_persisted(group_keys)),
              "is_persisted() is true after mark_persisted");

        rec.remove({"kv:0"});
        std::vector<bool> after_remove = rec.has(label, chunk_ids);
        CHECK(after_remove.size() == 3 && !after_remove[0] && after_remove[1] && after_remove[2],
              "has() reflects removal of kv:0");

        rec.shutdown();
    }

    {
        kv_pool::Record rec(db_path, true);
        constexpr int thread_count = 4;
        constexpr int keys_per_thread = 100;
        std::atomic<bool> submitting{true};
        std::thread reader([&]() {
            while (submitting.load(std::memory_order_acquire)) {
                rec.has("concurrent", {"0:0"});
                rec.sync();
            }
        });

        std::vector<std::thread> submitters;
        for (int thread = 0; thread < thread_count; ++thread) {
            submitters.emplace_back([&, thread]() {
                for (int key = 0; key < keys_per_thread; ++key) {
                    rec.submit("concurrent", {std::to_string(thread) + ':' + std::to_string(key)});
                }
            });
        }
        for (auto &submitter : submitters)
            submitter.join();
        submitting.store(false, std::memory_order_release);
        reader.join();
        rec.sync();

        bool all_submitted = true;
        for (int thread = 0; thread < thread_count; ++thread) {
            std::vector<std::string> ids;
            for (int key = 0; key < keys_per_thread; ++key) {
                ids.push_back(std::to_string(thread) + ':' + std::to_string(key));
            }
            all_submitted = all_submitted && all_true(rec.has("concurrent", ids));
        }
        CHECK(all_submitted, "concurrent submit/sync/has preserves all records");
    }

    {
        kv_pool::Record rec(db_path, false);
        std::vector<bool> reopened = rec.is_persisted({"kv:1", "kv:2"});
        CHECK(reopened.size() == 2 && reopened[0] && reopened[1],
              "persisted records survive reopen");
        CHECK(none_true(rec.is_persisted({"kv:0"})), "removed record absent after reopen");
    }

    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    if (g_failures == 0) {
        std::printf("\nrecord_test: PASS\n");
        return 0;
    }
    std::printf("\nrecord_test: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
