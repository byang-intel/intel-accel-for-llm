// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Lossy K4/V4 KV-cache codec. TurboQuant-style MSE quantization (Beta Lloyd-Max
// codebook + shared Haar rotation) reduces each bf16 head vector to a uint8
// centroid index plus one bf16 norm before entropy coding. It mirrors the role
// of data_shuffle -- invoked inside the compress/decompress pipeline -- except
// the serialized (indices || norms) payload replaces the raw tensor bytes.
//
// The codebook/rotation math and fused kernels follow
// reference.kvzip/kvzip_eval/tq2_kernel.cpp. The rotation GEMM is fused with the
// per-row quantize/dequantize: it borrows gemm_xeon's AMX brgemm microkernel
// (tinygemm_kernel + convert_weight_packed VNNI packing) so the intermediate
// rotated vectors stay cache-resident instead of doing a full DRAM round-trip.
// head_dim must be a multiple of 32 (fused AMX path only). Compiled with
// AMX/AVX-512 flags (see the per-source COMPILE_OPTIONS in CMakeLists.txt).

#include "lossy_k4v4.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>

// gemm_xeon (sglang CPU GEMM) building blocks: convert_weight_packed /
// tinygemm_kernel / can_use_brgemm / block_size_m,n / at::native::cpublas::brgemm.
#include "gemm.h"

#include "iaxl_common.h"

namespace kv_zip {
namespace {

constexpr int64_t K4V4_BITS = 4;   // "k4v4": 4-bit centroid index per coordinate
constexpr int64_t K4V4_SEED = 42;  // fixed seed -> encoder/decoder share one rotation

// ---- Regularized incomplete Beta I_x(a,b) and its inverse (Numerical Recipes) ----

double betacf(double a, double b, double x) {
    const int MAXIT = 200;
    const double EPS = 3.0e-12, FPMIN = 1.0e-300;
    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (std::fabs(d) < FPMIN)
        d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= MAXIT; ++m) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < FPMIN)
            d = FPMIN;
        c = 1.0 + aa / c;
        if (std::fabs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < FPMIN)
            d = FPMIN;
        c = 1.0 + aa / c;
        if (std::fabs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) <= EPS)
            break;
    }
    return h;
}

double betai(double a, double b, double x) {
    if (x <= 0.0)
        return 0.0;
    if (x >= 1.0)
        return 1.0;
    double lbt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) +
                 b * std::log(1.0 - x);
    double bt = std::exp(lbt);
    if (x < (a + 1.0) / (a + b + 2.0))
        return bt * betacf(a, b, x) / a;
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

double invbetai(double p, double a, double b) {
    const double EPS = 1.0e-10;
    double a1 = a - 1.0, b1 = b - 1.0, x, t, u, w;
    if (p <= 0.0)
        return 0.0;
    if (p >= 1.0)
        return 1.0;
    if (a >= 1.0 && b >= 1.0) {
        double pp = (p < 0.5) ? p : 1.0 - p;
        t = std::sqrt(-2.0 * std::log(pp));
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if (p < 0.5)
            x = -x;
        double al = (x * x - 3.0) / 6.0;
        double h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        w = (x * std::sqrt(al + h) / h) -
            (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        x = a / (a + b * std::exp(2.0 * w));
    } else {
        double lna = std::log(a / (a + b)), lnb = std::log(b / (a + b));
        t = std::exp(a * lna) / a;
        u = std::exp(b * lnb) / b;
        w = t + u;
        if (p < t / w)
            x = std::pow(a * w * p, 1.0 / a);
        else
            x = 1.0 - std::pow(b * w * (1.0 - p), 1.0 / b);
    }
    double afac = -std::lgamma(a) - std::lgamma(b) + std::lgamma(a + b);
    for (int j = 0; j < 10; ++j) {
        if (x <= 0.0 || x >= 1.0)
            return x;
        double err = betai(a, b, x) - p;
        t = std::exp(a1 * std::log(x) + b1 * std::log(1.0 - x) + afac);
        u = err / t;
        x -= (t = u / (1.0 - 0.5 * std::min(1.0, u * (a1 / x - b1 / (1.0 - x)))));
        if (x <= 0.0)
            x = 0.5 * (x + t);
        if (x >= 1.0)
            x = 0.5 * (x + t + 1.0);
        if (std::fabs(t) < EPS * x && j > 0)
            break;
    }
    return x;
}

// Finite-dimension Beta Lloyd-Max codebook (depends only on dim/bits). Produces
// `levels` centroids symmetric about 0 (ascending) and their `levels-1` midpoint
// decision boundaries.
void compute_codebook(int64_t d, int bits, std::vector<float> &centroids,
                      std::vector<float> &boundaries) {
    int levels = 1 << bits;
    double alpha = (d - 1) / 2.0;
    std::vector<double> c(levels);
    for (int i = 0; i < levels; ++i)
        c[i] = 2.0 * invbetai((i + 0.5) / levels, alpha, alpha) - 1.0;
    for (int it = 0; it < 200; ++it) {
        std::vector<double> bnd(levels + 1);
        bnd[0] = -1.0;
        bnd[levels] = 1.0;
        for (int j = 1; j < levels; ++j)
            bnd[j] = (c[j - 1] + c[j]) / 2.0;
        double maxdiff = 0.0;
        std::vector<double> upd(levels);
        for (int j = 0; j < levels; ++j) {
            double lt = (bnd[j] + 1.0) / 2.0, ut = (bnd[j + 1] + 1.0) / 2.0;
            double mass = betai(alpha, alpha, ut) - betai(alpha, alpha, lt);
            double fm = 0.5 * (betai(alpha + 1.0, alpha, ut) - betai(alpha + 1.0, alpha, lt));
            upd[j] = 2.0 * fm / mass - 1.0;
            maxdiff = std::max(maxdiff, std::fabs(upd[j] - c[j]));
        }
        c.swap(upd);
        if (maxdiff < 1e-12)
            break;
    }
    centroids.assign(levels, 0.f);
    for (int i = 0; i < levels; ++i)
        centroids[i] = static_cast<float>(c[i]);
    boundaries.assign(levels - 1, 0.f);
    for (int i = 0; i < levels - 1; ++i)
        boundaries[i] = static_cast<float>((c[i] + c[i + 1]) / 2.0);
}

// Shared Haar random rotation P (fixed seed -> encoder/decoder agree). Q from a
// sign-corrected QR of a Gaussian matrix, returned in bf16.
at::Tensor compute_rotation(int64_t d, int64_t seed) {
    at::Generator gen = at::detail::createCPUGenerator(static_cast<uint64_t>(seed));
    at::Tensor g = at::randn({d, d}, gen, at::TensorOptions().dtype(at::kFloat));
    auto qr = at::linalg_qr(g, "reduced");
    at::Tensor Q = std::get<0>(qr), R = std::get<1>(qr);
    at::Tensor diag = at::diagonal(R);
    at::Tensor sign = at::where(diag < 0, at::full_like(diag, -1.0f), at::full_like(diag, 1.0f));
    return (Q * sign.unsqueeze(0)).to(at::kBFloat16);
}

// (codebook + P/Pt) cached by (dim, bits, seed).
struct Params {
    at::Tensor centroids;   // [levels]    fp32
    at::Tensor boundaries;  // [levels-1]  fp32
    at::Tensor rotation;    // [d, d] bf16 (P)
    at::Tensor rotation_t;  // [d, d] bf16 (Pt)
    // VNNI-prepacked rotations for the fused AMX path.
    // brgemm(A, convert_weight_packed(W)) computes A @ W^T.
    at::Tensor packed_P;   // convert_weight_packed(P)  -> pack:   brgemm(x, packed_P)  = x @ P^T
    at::Tensor packed_Pt;  // convert_weight_packed(Pt) -> unpack: brgemm(w, packed_Pt) = w @ P
};

std::mutex g_params_mutex;
std::map<std::tuple<int64_t, int, int64_t>, Params> g_params_cache;

const Params &get_params(int64_t d, int bits, int64_t seed) {
    auto key = std::make_tuple(d, bits, seed);
    // Lock-free fast path: each thread remembers its last lookup.
    thread_local std::tuple<int64_t, int, int64_t> last_key{-1, -1, -1};
    thread_local const Params *last_ptr = nullptr;
    if (last_ptr && key == last_key)
        return *last_ptr;
    std::lock_guard<std::mutex> lock(g_params_mutex);
    auto it = g_params_cache.find(key);
    if (it == g_params_cache.end()) {
        std::vector<float> cvec, bvec;
        compute_codebook(d, bits, cvec, bvec);
        Params p;
        p.centroids = at::from_blob(cvec.data(), {(int64_t)cvec.size()},
                                    at::TensorOptions().dtype(at::kFloat))
                          .clone();
        p.boundaries = at::from_blob(bvec.data(), {(int64_t)bvec.size()},
                                     at::TensorOptions().dtype(at::kFloat))
                           .clone();
        at::Tensor P = compute_rotation(d, seed);
        p.rotation = P.contiguous();
        p.rotation_t = P.t().contiguous();
        // Fused AMX path only: convert_weight_packed needs K = d % 32 == 0.
        IAXL_CHECK(d % 32 == 0, "lossy_k4v4: head_dim must be a multiple of 32");
        at::Tensor Pc = p.rotation.clone();   // convert_weight_packed takes Tensor&
        at::Tensor Ptc = p.rotation_t.clone();
        p.packed_P = convert_weight_packed(Pc);
        p.packed_Pt = convert_weight_packed(Ptc);
        it = g_params_cache.emplace(key, std::move(p)).first;
    }
    last_key = key;
    last_ptr = &it->second;
    return it->second;
}

// ---- Fused AMX / AVX-512 quantize-dequantize kernels ----
// Ported from reference.kvzip/kvzip_eval/tq2_kernel.cpp. The codebook is
// symmetric about 0, so the bucketize compares |y| against only the positive-half
// boundaries and restores the index by sign, halving the comparisons.

constexpr int64_t PACK_MAX_D = 1024; // per-thread fp32 buffer bound

// Fused quantize (d % 32 == 0): z = A @ P^T via tinygemm/AMX brgemm, immediately
// followed by per-row norm + symmetric bucketize. Each M-block's z stays in a
// cache-resident buffer instead of a full DRAM round-trip. packed_P = convert_weight_packed(P).
void pack_fused(const at::BFloat16 *x, uint8_t *idx_out, at::BFloat16 *norm_out, int64_t rows,
                int64_t d, const at::BFloat16 *packed_P, const float *boundaries, int nbound) {
    constexpr int64_t BM = block_size_m(); // 32 (gemm_xeon AMX block)
    constexpr int64_t BN = block_size_n(); // 32
    const int64_t MB = (rows + BM - 1) / BM;
    const int64_t NB = d / BN; // d % 32 == 0 -> exact
    const int half = nbound / 2;
    const float *posb = boundaries + half + 1;
#pragma omp parallel
    {
        std::vector<at::BFloat16> zrow(BM * d); // per-thread cache-resident z block
        alignas(64) float Ctmp[BM * BN];        // tinygemm/brgemm fp32 accumulator scratch
        alignas(64) float zbuf[PACK_MAX_D];     // per-row fp32 reused across the two passes
        bool used_brg = false;
#pragma omp for schedule(static)
        for (int64_t mb = 0; mb < MB; ++mb) {
            const int64_t m0 = mb * BM;
            const int64_t msz = std::min(rows - m0, BM);
            const at::BFloat16 *A = x + m0 * d;
            const int64_t lda = d;
            // z_block = A @ P^T, one N-block at a time (M>4 -> AMX brgemm, else tinygemm).
            const bool brg = can_use_brgemm<at::BFloat16>(static_cast<int>(msz));
            for (int64_t nb = 0; nb < NB; ++nb) {
                const int64_t n0 = nb * BN;
                tinygemm_kernel<at::BFloat16>(A, packed_P + n0 * d, zrow.data() + n0, Ctmp, msz, BN,
                                              d, lda, BN, d, brg);
            }
            used_brg |= brg;
            // epilogue: per-row norm + symmetric bucketize (reads bf16 zrow, reuses fp32 zbuf).
            for (int64_t m = 0; m < msz; ++m) {
                const at::BFloat16 *zr = zrow.data() + m * d;
                uint8_t *ir = idx_out + (m0 + m) * d;
                __m512 acc = _mm512_setzero_ps();
                for (int64_t c = 0; c < d; c += 16) {
                    __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(zr + c));
                    __m512 zf =
                        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
                    _mm512_store_ps(zbuf + c, zf);
                    acc = _mm512_fmadd_ps(zf, zf, acc);
                }
                const float nrm = std::sqrt(_mm512_reduce_add_ps(acc));
                const __m512 invv = _mm512_set1_ps(nrm > 0.f ? 1.f / nrm : 0.f);
                const __m512i one = _mm512_set1_epi32(1);
                const __m512 zero = _mm512_setzero_ps();
                const __m512i vhalf = _mm512_set1_epi32(half);
                const __m512i vhalf1 = _mm512_set1_epi32(half + 1);
                for (int64_t c = 0; c < d; c += 16) {
                    __m512 y = _mm512_mul_ps(_mm512_load_ps(zbuf + c), invv);
                    __m512 a = _mm512_abs_ps(y);
                    __m512i off = _mm512_setzero_si512();
                    for (int k = 0; k < half; ++k) {
                        __mmask16 gt = _mm512_cmp_ps_mask(a, _mm512_set1_ps(posb[k]), _CMP_GT_OQ);
                        off = _mm512_mask_add_epi32(off, gt, off, one);
                    }
                    __mmask16 pos = _mm512_cmp_ps_mask(y, zero, _CMP_GT_OQ);
                    __m512i vi = _mm512_mask_blend_epi32(pos, _mm512_sub_epi32(vhalf, off),
                                                         _mm512_add_epi32(vhalf1, off));
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(ir + c), _mm512_cvtepi32_epi8(vi));
                }
                norm_out[m0 + m] = at::BFloat16(nrm);
            }
        }
        if (used_brg)
            at::native::cpublas::brgemm_release(); // release this thread's AMX tile config
    }
}

// Fused dequantize (d % 32 == 0): rebuild w = norm * centroid[idx] in a cache-resident
// buffer, then xhat = w @ P via tinygemm/brgemm writing straight to the output.
// packed_Pt = convert_weight_packed(P^T). xhat_strideM is the output row stride.
void unpack_fused(const uint8_t *idx, const at::BFloat16 *norms, at::BFloat16 *xhat,
                  int64_t xhat_strideM, int64_t rows, int64_t d, const float *centroids, int levels,
                  const at::BFloat16 *packed_Pt) {
    constexpr int64_t BM = block_size_m();
    constexpr int64_t BN = block_size_n();
    const int64_t MB = (rows + BM - 1) / BM;
    const int64_t NB = d / BN;
    alignas(64) float cent16[16];
    for (int i = 0; i < 16; ++i)
        cent16[i] = centroids[i < levels ? i : levels - 1];
#pragma omp parallel
    {
        std::vector<at::BFloat16> wrow(BM * d); // per-thread cache-resident rebuilt w
        alignas(64) float Ctmp[BM * BN];
        const __m512 cvec = _mm512_load_ps(cent16);
        bool used_brg = false;
#pragma omp for schedule(static)
        for (int64_t mb = 0; mb < MB; ++mb) {
            const int64_t m0 = mb * BM;
            const int64_t msz = std::min(rows - m0, BM);
            // rebuild w[m,c] = norms[m] * centroid[idx[m,c]] (vpermps select, no gather).
            for (int64_t m = 0; m < msz; ++m) {
                const uint8_t *ir = idx + (m0 + m) * d;
                at::BFloat16 *wr = wrow.data() + m * d;
                const __m512 nv = _mm512_set1_ps(static_cast<float>(norms[m0 + m]));
                for (int64_t c = 0; c < d; c += 16) {
                    __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ir + c));
                    __m512i vidx = _mm512_cvtepu8_epi32(raw);
                    __m512 yhat = _mm512_permutexvar_ps(vidx, cvec);
                    __m512 wv = _mm512_mul_ps(yhat, nv);
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(wr + c),
                                        (__m256i)_mm512_cvtneps_pbh(wv));
                }
            }
            // xhat_block = w @ P, one N-block at a time, writing the final output directly.
            const bool brg = can_use_brgemm<at::BFloat16>(static_cast<int>(msz));
            for (int64_t nb = 0; nb < NB; ++nb) {
                const int64_t n0 = nb * BN;
                tinygemm_kernel<at::BFloat16>(wrow.data(), packed_Pt + n0 * d,
                                              xhat + m0 * xhat_strideM + n0, Ctmp, msz, BN, d, d, BN,
                                              xhat_strideM, brg);
            }
            used_brg |= brg;
        }
        if (used_brg)
            at::native::cpublas::brgemm_release();
    }
}

// pack: x2 [rows,d] bf16 -> idx_out [rows,d] uint8 + norm_out [rows,1] bf16.
// idx_out/norm_out are from_blob views into the serialized payload, so the fused
// kernel writes the indices/norms directly into the payload (no extra staging).
void pack_into(const at::Tensor &x2, at::Tensor &idx_out, at::Tensor &norm_out,
               const Params &prm) {
    const int64_t rows = x2.size(0), d = x2.size(1);
    at::Tensor bnd = prm.boundaries.to(at::kFloat).contiguous();
    const int nbound = static_cast<int>(bnd.numel());
    pack_fused(x2.data_ptr<at::BFloat16>(), idx_out.data_ptr<uint8_t>(),
               norm_out.data_ptr<at::BFloat16>(), rows, d, prm.packed_P.data_ptr<at::BFloat16>(),
               bnd.data_ptr<float>(), nbound);
}

// unpack: idx [rows,d] uint8 + norms [rows,1] bf16 -> out [rows,d] bf16 (tensor storage).
void unpack_into(const at::Tensor &idx, const at::Tensor &norms, const Params &prm,
                 at::Tensor &out) {
    const int64_t rows = idx.size(0), d = idx.size(1);
    at::Tensor cent = prm.centroids.to(at::kFloat).contiguous();
    const int levels = static_cast<int>(cent.numel());
    unpack_fused(idx.data_ptr<uint8_t>(), norms.data_ptr<at::BFloat16>(),
                 out.data_ptr<at::BFloat16>(), d, rows, d, cent.data_ptr<float>(), levels,
                 prm.packed_Pt.data_ptr<at::BFloat16>());
}

} // namespace

bool lossy_k4v4_applicable(const torch::Tensor &t) {
    if (!lossy_k4v4_enabled())
        return false;
    if (t.scalar_type() != at::kBFloat16 || t.dim() == 0)
        return false;
    const int64_t d = t.size(-1);
    return d >= 32 && d <= 1024 && (d % 32 == 0);
}

char *lossy_k4v4_serialize(const torch::Tensor &t, size_t *out_size) {
    at::NoGradGuard no_grad;
    const int64_t d = t.size(-1);
    const int64_t rows = t.numel() / d;
    const Params &prm = get_params(d, static_cast<int>(K4V4_BITS), K4V4_SEED);

    const size_t idx_bytes = static_cast<size_t>(rows) * static_cast<size_t>(d);
    const size_t total = idx_bytes + static_cast<size_t>(rows) * sizeof(at::BFloat16);
    char *buf = static_cast<char *>(malloc(total));
    IAXL_CHECK(buf != nullptr, "lossy_k4v4: payload allocation failed");

    at::Tensor idx_view =
        at::from_blob(buf, {rows, d}, at::TensorOptions().dtype(at::kByte));
    at::Tensor norm_view =
        at::from_blob(buf + idx_bytes, {rows, 1}, at::TensorOptions().dtype(at::kBFloat16));
    at::Tensor x2 = t.reshape({rows, d}); // bf16 view; t is contiguous

    pack_into(x2, idx_view, norm_view, prm);
    *out_size = total;
    return buf;
}

void lossy_k4v4_deserialize(const void *src, size_t src_size, const torch::Tensor &t) {
    at::NoGradGuard no_grad;
    const int64_t d = t.size(-1);
    const int64_t rows = t.numel() / d;
    const size_t idx_bytes = static_cast<size_t>(rows) * static_cast<size_t>(d);
    IAXL_CHECK(src_size == idx_bytes + static_cast<size_t>(rows) * sizeof(at::BFloat16),
               "lossy_k4v4: serialized payload size mismatch");
    const Params &prm = get_params(d, static_cast<int>(K4V4_BITS), K4V4_SEED);

    char *base = static_cast<char *>(const_cast<void *>(src));
    at::Tensor idx_view =
        at::from_blob(base, {rows, d}, at::TensorOptions().dtype(at::kByte));
    at::Tensor norm_view =
        at::from_blob(base + idx_bytes, {rows, 1}, at::TensorOptions().dtype(at::kBFloat16));
    at::Tensor out_view =
        at::from_blob(t.data_ptr(), {rows, d}, at::TensorOptions().dtype(at::kBFloat16));

    unpack_into(idx_view, norm_view, prm, out_view);
}

} // namespace kv_zip
