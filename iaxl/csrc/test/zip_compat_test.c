// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>

#include "cpu_zip.h"
#include "env.h"
#include "qat_zip.h"

static int raw_deflate(const unsigned char *src, int src_len, unsigned char *dst, int *dst_len) {
    z_stream stream = {0};
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK)
        return -1;
    stream.next_in = (Bytef *)src;
    stream.avail_in = (uInt)src_len;
    stream.next_out = dst;
    stream.avail_out = (uInt)*dst_len;
    int rc = deflate(&stream, Z_FINISH);
    *dst_len = (int)stream.total_out;
    deflateEnd(&stream);
    return rc == Z_STREAM_END ? 0 : -1;
}

static int raw_inflate(const unsigned char *src, int src_len, unsigned char *dst, int *dst_len) {
    z_stream stream = {0};
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
        return -1;
    stream.next_in = (Bytef *)src;
    stream.avail_in = (uInt)src_len;
    stream.next_out = dst;
    stream.avail_out = (uInt)*dst_len;
    int rc = inflate(&stream, Z_FINISH);
    *dst_len = (int)stream.total_out;
    inflateEnd(&stream);
    return rc == Z_STREAM_END ? 0 : -1;
}

static int verify(const char *name, const unsigned char *expected, int expected_len, void *actual,
                  int actual_len) {
    if (actual_len != expected_len || memcmp(actual, expected, (size_t)expected_len) != 0) {
        fprintf(stderr, "[compat] %s FAILED: expected %d bytes, got %d\n", name, expected_len,
                actual_len);
        return -1;
    }
    printf("[compat] %s passed\n", name);
    return 0;
}

int main(int argc, char **argv) {
    static const int input_len = 64 * 1024;
    const char *devices = argc > 1 ? argv[1] : NULL;
    int status = 1;
    int failures = 0;
    unsigned char *input = NULL;
    unsigned char *qat_data = NULL;
    unsigned char *cpu_data = NULL;
    unsigned char *raw_data = NULL;

    if (devices)
        setenv("IAXL_QAT_DEVICES", devices, 1);
    setenv("IAXL_CPU_ZIP_THREADS", "1", 1);
    setenv("IAXL_ZIP_SRC_CAP", "262144", 1);
    setenv("IAXL_ZIP_DST_CAP", "262144", 1);
    envs_init();

    input = malloc((size_t)input_len);
    if (!input)
        goto out;
    for (int i = 0; i < input_len; i++)
        input[i] = (unsigned char)(i % 251);

    if (qat_zip_init() != 0 || cpu_zip_init() != 0) {
        fprintf(stderr, "[compat] backend initialization failed\n");
        goto out;
    }

    void *output;
    int output_len;
    int rc = qat_zip_compress(0, input, input_len);
    if (rc != 0) {
        fprintf(stderr, "[compat] QAT compression submission failed: %d\n", rc);
        goto out;
    }
    rc = qat_zip_wait(0, &output, &output_len);
    if (rc != 0 || output_len <= 0) {
        fprintf(stderr, "[compat] QAT compression completion failed: status=%d length=%d\n", rc,
                output_len);
        goto out;
    }
    if (!output) {
        fprintf(stderr, "[compat] QAT compression failed\n");
        goto out;
    }
    qat_data = malloc((size_t)output_len);
    if (!qat_data)
        goto out;
    memcpy(qat_data, output, (size_t)output_len);
    int qat_len = output_len;

    rc = cpu_zip_decompress(0, qat_data, qat_len);
    if (rc != 0) {
        fprintf(stderr, "[compat] QAT compress -> CPU decompress rejected: %d\n", rc);
        failures++;
    } else {
        rc = cpu_zip_wait(0, &output, &output_len);
        if (rc != 0 ||
            verify("QAT compress -> CPU decompress", input, input_len, output, output_len) != 0)
            failures++;
    }

    rc = cpu_zip_compress(0, input, input_len);
    if (rc != 0 || (rc = cpu_zip_wait(0, &output, &output_len)) != 0 || output_len <= 0) {
        fprintf(stderr, "[compat] CPU compression failed: status=%d length=%d\n", rc, output_len);
        goto out;
    }
    cpu_data = malloc((size_t)output_len);
    if (!cpu_data)
        goto out;
    memcpy(cpu_data, output, (size_t)output_len);
    int cpu_len = output_len;

    rc = qat_zip_decompress(0, cpu_data, cpu_len);
    if (rc != 0) {
        fprintf(stderr, "[compat] CPU compress -> QAT decompress submission failed: %d\n", rc);
        failures++;
    } else {
        rc = qat_zip_wait(0, &output, &output_len);
        if (rc != 0) {
            fprintf(stderr, "[compat] CPU compress -> QAT decompress completion failed: %d\n", rc);
            failures++;
        } else if (verify("CPU compress -> QAT decompress", input, input_len, output, output_len) !=
                   0) {
            failures++;
        }
    }

    if (failures == 0) {
        printf("[compat] QAT and CPU compressed streams are mutually compatible\n");
    } else {
        fprintf(stderr, "[compat] QAT and CPU compressed streams are not mutually compatible\n");
    }

    int raw_cap = envs.IAXL_ZIP_SRC_CAP > envs.IAXL_ZIP_DST_CAP ? envs.IAXL_ZIP_SRC_CAP
                                                                : envs.IAXL_ZIP_DST_CAP;
    raw_data = malloc((size_t)raw_cap);
    if (!raw_data)
        goto out;
    int raw_len = envs.IAXL_ZIP_SRC_CAP;
    if (raw_inflate(qat_data, qat_len, raw_data, &raw_len) != 0 ||
        verify("QAT compress -> raw zlib inflate", input, input_len, raw_data, raw_len) != 0)
        goto out;

    raw_len = envs.IAXL_ZIP_DST_CAP;
    if (raw_deflate(input, input_len, raw_data, &raw_len) != 0) {
        fprintf(stderr, "[compat] raw zlib deflate failed\n");
        goto out;
    }
    rc = qat_zip_decompress(0, raw_data, raw_len);
    if (rc != 0 || (rc = qat_zip_wait(0, &output, &output_len)) != 0 ||
        verify("raw zlib deflate -> QAT decompress", input, input_len, output, output_len) != 0) {
        fprintf(stderr, "[compat] raw zlib -> QAT failed: %d\n", rc);
        goto out;
    }
    printf("[compat] QAT uses raw DEFLATE; CPU can interoperate by using zlib raw mode\n");
    if (failures == 0)
        status = 0;

out:
    free(raw_data);
    free(cpu_data);
    free(qat_data);
    free(input);
    cpu_zip_shutdown();
    qat_zip_shutdown();
    return status;
}