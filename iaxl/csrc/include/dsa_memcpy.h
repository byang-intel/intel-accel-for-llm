// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef IAXL_DSA_MEMCPY_H
#define IAXL_DSA_MEMCPY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int dsa_memcpy(void *dest, const void *src, size_t n);
int dsa_memcpy_batch(void *const dest[], const void *const src[], const size_t n[], size_t count);

/*
 * Lanes let H2D and D2H traffic run full duplex: each lane owns its own subset of the WQs
 * listed in $IAXL_DSA_WQS (lane L takes WQ L, L+2, ...; with a single WQ both lanes share it)
 * and its own ticket sequence, so waiting on one lane never waits for the other.
 */
#define DSA_LANE_H2D 0u
#define DSA_LANE_D2H 1u
#define DSA_LANES 2u

/*
 * Async variant: returns once every batch is submitted, leaving a few batches per WQ in
 * flight so back-to-back calls keep the engine busy. Returns a ticket (> 0) for this call,
 * 0 if count == 0, or -1 on error.
 *
 * dsa_memcpy_batch_wait(ticket) blocks until that call and every earlier async call on the
 * same lane has completed (DSA_TICKET_ALL drains every lane) and returns -1 if any of them
 * failed. The two functions are safe to call from different threads: a waiter never blocks a
 * submitter and only waits for the tickets it asked for. dsa_memcpy_batch() must not run while
 * async batches are in flight.
 */
int64_t dsa_memcpy_batch_async(unsigned lane, void *const dest[], const void *const src[],
                               const size_t n[], size_t count);
int dsa_memcpy_batch_wait(uint64_t ticket);
#define DSA_TICKET_ALL UINT64_MAX

#ifdef __cplusplus
}
#endif

#endif
