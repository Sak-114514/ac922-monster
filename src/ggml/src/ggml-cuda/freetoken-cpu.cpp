// SPDX-License-Identifier: Apache-2.0
// NVFP4 arithmetic adapted from FreeToken cpu_moe_ext.cpp; see licenses/LICENSE-freetoken.
// Source SHA256: f25698c3a9a9741f4ed3e7460a2af7558cb42111531f04de1472dd8835cef8d1
// Native adaptation removes Python/CUDA dependencies and limits execution to NVFP4.
#include "freetoken-cpu.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define FT_CPU_VNNI 1
#endif

namespace {
using bf16_t = uint16_t;
inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // round-to-nearest-even
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

alignas(16) const int8_t kE2M1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

// e4m3 (OCP "fn": finite, max-normal 448, exp bias 7). Decoded into a 256-entry LUT.
inline float e4m3_decode(uint8_t v) {
  const float sign = (v & 0x80u) ? -1.0f : 1.0f;
  const uint32_t exp = (v >> 3) & 0xFu;
  const uint32_t man = v & 0x7u;
  if (exp == 0) return sign * (man / 8.0f) * 0.015625f;  // 2^(1-7) = 2^-6
  return sign * (1.0f + man / 8.0f) * std::ldexp(1.0f, (int)exp - 7);
}

[[maybe_unused]] float dot_nvfp4_i8_scalar(const uint8_t* packed, const uint8_t* scale,
                          float global, const int8_t* asi8, int K, const float* e4m3,
                          const float* asb) {
  float acc = 0.0f;
  const int nb = K / 16;
  for (int b = 0; b < nb; ++b) {
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16;       // even(8)
    const int8_t* ao = ae + 8;                      // odd(8)
    int isum = 0;
    for (int j = 0; j < 8; ++j) {
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j];
      isum += (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    }
    acc += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return acc * (0.5f * global);
}


#ifdef FT_CPU_VNNI
static int nvfp4_pf_blocks() { return -1; }
__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
static inline __m512 nvfp4_i8_grp4(const uint8_t* packed, const uint8_t* scale,
                                   const int8_t* asi8, const float* e4m3, const float* asb,
                                   int b, __m512i lut, __m512i idx, __m512i mask0F,
                                   __m512i idxsc) {
  const __mmask64 hi_half = 0xFF00FF00FF00FF00ULL;  // bytes 8-15 of each 128b lane -> hi nibbles
  __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + (size_t)b * 8));
  __m512i src = _mm512_permutexvar_epi64(idx, _mm512_castsi256_si512(raw));
  __m512i lo = _mm512_and_si512(src, mask0F);
  __m512i hi = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask0F);
  __m512i comb = _mm512_mask_blend_epi8(hi_half, lo, hi);
  __m512i w = _mm512_shuffle_epi8(lut, comb);  // 64 int8 weights (e2m1*2)
  // u8*s8 sign trick without _mm512_sign_epi8: aw=|w|, sa = (w<0 ? -a : a) (a moot at w==0)
  __m512i a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(asi8 + (size_t)b * 16));
  __m512i aw = _mm512_abs_epi8(w);
  __mmask64 neg = _mm512_movepi8_mask(w);
  __m512i sa = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
  __m512i di = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sa);  // 16 int32 (groups of 4)
  // int32[0..3]->blk b, [4..7]->b+1, [8..11]->b+2, [12..15]->b+3. The 4 block scales
  // (e4m3 LUT x per-block act scale) are computed vectorized: a 4-byte load + epu8->epi32
  // widen + one 4-lane LUT gather + one mul replaces 8 scalar loads + 4 scalar muls +
  // a set_ps assembly, which otherwise dominates the per-group op count.
  int sc_raw;
  memcpy(&sc_raw, scale + b, 4);
  __m128i sc4 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(sc_raw));
  __m128 s4 = _mm_mul_ps(_mm_i32gather_ps(e4m3, sc4, 4), _mm_loadu_ps(asb + b));
  __m512 scv = _mm512_permutexvar_ps(idxsc, _mm512_castps128_ps512(s4));
  return _mm512_mul_ps(_mm512_cvtepi32_ps(di), scv);
}

__attribute__((target("avx512f,avx512bw,avx512vnni,avx2")))
float dot_nvfp4_i8_avx512vnni(const uint8_t* packed, const uint8_t* scale, float global,
                              const int8_t* asi8, int K, const float* e4m3, const float* asb) {
  const __m512i lut = _mm512_broadcast_i32x4(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(kE2M1x2)));
  const __m512i idx = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);  // block i -> 128b lane i
  const __m512i mask0F = _mm512_set1_epi8(0x0F);
  const __m512i idxsc = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
  const int nb = K / 16;
  // Built-in default: min(4 KB, 2 rows) -- see nvfp4_pf_blocks(). An explicitly set
  // env value is used verbatim so a per-machine sweep can reach operating points past
  // the conservative default.
  const int pfb = nvfp4_pf_blocks();
  const int pf = (pfb < 0) ? std::min(512, 2 * nb) : pfb;
  int b = 0;
  // Four independent 4-block groups per iter -> four accumulator chains hide the ~10-op
  // decode->dpbusd->scale latency (1-2 chains leave the loop latency-bound: the per-core
  // rate sat at ~60% of the core's achievable DRAM stream rate).
  for (; b + 16 <= nb; b += 16) {
    // Prefetch the weight stream ahead: the interleaved decode lowers the L1-miss
    // concurrency the HW prefetcher sustains on its own (134 -> 167 GB/s on Emerald Rapids).
    if (pf > 0) {
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8),
                   _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(packed + ((size_t)b + (size_t)pf) * 8 + 64),
                   _MM_HINT_T0);
    }
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
    acc1 = _mm512_add_ps(acc1, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 4, lut, idx,
                                             mask0F, idxsc));
    acc2 = _mm512_add_ps(acc2, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 8, lut, idx,
                                             mask0F, idxsc));
    acc3 = _mm512_add_ps(acc3, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b + 12, lut, idx,
                                             mask0F, idxsc));
  }
  for (; b + 4 <= nb; b += 4)
    acc0 = _mm512_add_ps(acc0, nvfp4_i8_grp4(packed, scale, asi8, e4m3, asb, b, lut, idx,
                                             mask0F, idxsc));
  float s = _mm512_reduce_add_ps(
      _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
  for (; b < nb; ++b) {  // tail (<4 remaining 16-K blocks)
    const uint8_t* pk = packed + (size_t)b * 8;
    const int8_t* ae = asi8 + (size_t)b * 16; const int8_t* ao = ae + 8;
    int isum = 0;
    for (int j = 0; j < 8; ++j)
      isum += (int)kE2M1x2[pk[j] & 0xF] * (int)ae[j] + (int)kE2M1x2[pk[j] >> 4] * (int)ao[j];
    s += (e4m3[scale[b]] * asb[b]) * (float)isum;
  }
  return s * (0.5f * global);
}

#endif

void pause_worker() {
#ifdef FT_CPU_VNNI
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

void quantize(const uint16_t * x, int n, int8_t * q, float * scales) {
    for (int b = 0; b < n / 16; ++b) {
        float values[16], amax = 0;
        for (int j = 0; j < 16; ++j) {
            values[j] = bf16_to_f32(x[16*b + j]);
            amax = std::max(amax, std::abs(values[j]));
        }
        const float scale = amax > 0 ? amax / 127.0f : 1.0f;
        scales[b] = scale;
        const float inv = 1.0f / scale;
        for (int j = 0; j < 16; ++j) {
            const int v = std::max(-127, std::min(127, (int) std::lround(values[j] * inv)));
            q[16*b + j/2 + (j%2)*8] = (int8_t) v;
        }
    }
}
} // namespace

void freetoken_unpack_nvfp4(const void * gguf_data, int64_t rows, int64_t columns,
                           uint8_t * packed, uint8_t * scales) {
    if (!gguf_data || !packed || !scales || rows <= 0 || columns <= 0 || columns % 64
            || rows > INT64_MAX / columns) {
        throw std::invalid_argument("invalid GGUF NVFP4 bank");
    }
    const auto * src = static_cast<const uint8_t *>(gguf_data);
    // GGUF stores four scale bytes and 32 quant bytes per block of 64 weights.
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t b = 0; b < columns/64; ++b) {
            const uint8_t * block = src + (row*(columns/64) + b)*36;
            for (int g = 0; g < 4; ++g) {
                scales[row*(columns/16) + b*4 + g] = block[g];
                const uint8_t * q = block + 4 + g*8;
                uint8_t * dst = packed + row*(columns/2) + b*32 + g*8;
                for (int j = 0; j < 8; ++j) {
                    const int shift = j < 4 ? 0 : 4;
                    const int lo = (q[(2*j)%8] >> shift) & 15;
                    const int hi = (q[(2*j+1)%8] >> shift) & 15;
                    dst[j] = (uint8_t) (lo | (hi << 4));
                }
            }
        }
    }
}

struct freetoken_cpu_executor::impl {
    using dot_fn = float (*)(const uint8_t *, const uint8_t *, float, const int8_t *, int,
                            const float *, const float *);
    int H, I, E, K, max_tokens, n_threads;
    std::vector<freetoken_cpu_bank> banks;
    std::vector<int> cores;
    float e4m3[256];
    dot_fn dot = dot_nvfp4_i8_scalar;
    const char * isa_name = "scalar-w4a8";
    std::vector<uint16_t> intermediate;
    std::vector<int8_t> x_quant, g_quant;
    std::vector<float> x_scale, g_scale;
    std::vector<std::thread> workers;
    std::atomic<bool> stop{false};
    std::atomic_flag running = ATOMIC_FLAG_INIT;
    alignas(64) std::atomic<uint64_t> generation{0};
    alignas(64) std::atomic<int> done{0};
    alignas(64) std::atomic<int> next_gate{0};
    alignas(64) std::atomic<int> next_prep{0};
    alignas(64) std::atomic<int> next_down{0};
    alignas(64) std::atomic<int> barrier_count{0};
    alignas(64) std::atomic<int> barrier_epoch{0};
    std::mutex mutex;
    std::condition_variable wake;
    int sleeping = 0;
    const bool skip_empty = !std::getenv("GGML_FT_CPU_SKIP_EMPTY") || std::strcmp(std::getenv("GGML_FT_CPU_SKIP_EMPTY"),"0") != 0;
    const bool fused_prep = std::getenv("GGML_FT_CPU_FUSED_PREP") && std::strcmp(std::getenv("GGML_FT_CPU_FUSED_PREP"),"1") == 0;
    struct job_t {
        int layer, tokens;
        const int32_t * ids;
        const float * weights;
        uint16_t * y;
        const int8_t * xq;
        const float * xs;
    } job{};

    impl(int h, int i, int e, int k, int tokens, int threads,
         std::vector<freetoken_cpu_bank> b, std::vector<int> c) :
        H(h), I(i), E(e), K(k), max_tokens(tokens), n_threads(threads), banks(std::move(b)), cores(std::move(c)) {
        if (H <= 0 || I <= 0 || H % 64 || I % 64 || H > 65536 || I > 65536 || E <= 0
                || K <= 0 || K > E || K > 64 || tokens < 1 || tokens > 4 || threads < 1 || threads > 256
                || banks.empty() || banks.size() > 128) {
            throw std::invalid_argument("invalid FreeToken CPU decode dimensions");
        }
        for (const auto & bank : banks) {
            if (!bank.gate_up || !bank.down || !bank.gate_up_scale || !bank.down_scale
                    || !bank.gate_up_global || !bank.down_global) {
                throw std::invalid_argument("missing FreeToken CPU expert bank");
            }
        }
#if defined(__linux__)
        cpu_set_t allowed;
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0 && !cores.empty()) {
            throw std::runtime_error("cannot read CPU affinity");
        }
        for (int core : cores) {
            if (core < 0 || core >= CPU_SETSIZE || !CPU_ISSET(core, &allowed)) {
                throw std::invalid_argument("FreeToken worker core is not available");
            }
        }
#endif
        for (int v = 0; v < 256; ++v) {
            e4m3[v] = e4m3_decode((uint8_t) v);
        }
#ifdef FT_CPU_VNNI
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
                && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx2")) {
            dot = dot_nvfp4_i8_avx512vnni;
            isa_name = "avx512vnni-w4a8";
        }
#endif
        intermediate.resize((size_t) tokens*K*I);
        g_quant.resize(intermediate.size());
        g_scale.resize(intermediate.size()/16);
        x_quant.resize((size_t) tokens*H);
        x_scale.resize(x_quant.size()/16);
        try {
            for (int t = 0; t < n_threads; ++t) {
                workers.emplace_back([this, t] { worker(t); });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop.store(true, std::memory_order_release);
        }
        wake.notify_all();
        for (auto & worker : workers) {
            if (worker.joinable()) { worker.join(); }
        }
    }
    ~impl() { shutdown(); }

    void barrier() {
        const int epoch = barrier_epoch.load(std::memory_order_acquire);
        if (barrier_count.fetch_add(1, std::memory_order_acq_rel) == n_threads - 1) {
            barrier_count.store(0, std::memory_order_relaxed);
            barrier_epoch.fetch_add(1, std::memory_order_release);
        } else {
            while (barrier_epoch.load(std::memory_order_acquire) == epoch) { pause_worker(); }
        }
    }

    void compute() {
        const auto & bank = banks[job.layer];
        const int tiles = I / 32;
        for (int p; (p = next_gate.fetch_add(1, std::memory_order_relaxed)) < job.tokens*K*tiles;) {
            const int route = p / tiles, token = route / K, expert = job.ids[route];
            if (expert < 0 || expert >= E) { continue; }
            for (int i = (p%tiles)*32; i < (p%tiles + 1)*32; ++i) {
                const size_t row = (size_t) expert*2*I + i;
                const auto gemv = [&](size_t r) {
                    return dot(bank.gate_up + r*(H/2), bank.gate_up_scale + r*(H/16),
                               fp16_to_f32(bank.gate_up_global[r]), job.xq + token*H, H, e4m3,
                               job.xs + token*(H/16));
                };
                const float gate = gemv(row), up = gemv(row + I);
                intermediate[(size_t) route*I + i] = f32_to_bf16(gate / (1.0f + std::exp(-gate)) * up);
            }
            if (fused_prep) {
                const size_t offset = (size_t) route*I + (p%tiles)*32;
                quantize(intermediate.data() + offset, 32, g_quant.data() + offset, g_scale.data() + offset/16);
            }
        }
        if (!fused_prep) {
            barrier();
            for (int route; (route = next_prep.fetch_add(1, std::memory_order_relaxed)) < job.tokens*K;) {
                if (job.ids[route] < 0 || job.ids[route] >= E) { continue; }
                quantize(intermediate.data() + (size_t) route*I, I,
                         g_quant.data() + (size_t) route*I, g_scale.data() + (size_t) route*(I/16));
            }
        }
        barrier();
        for (int p; (p = next_down.fetch_add(1, std::memory_order_relaxed)) < job.tokens*(H/32);) {
            const int token = p/(H/32), begin = (p%(H/32))*32;
            for (int h = begin; h < begin + 32; ++h) {
                float sum = 0;
                for (int k = 0; k < K; ++k) {
                    const int route = token*K + k, expert = job.ids[route];
                    if (expert < 0 || expert >= E) { continue; }
                    const size_t row = (size_t) expert*H + h;
                    sum += dot(bank.down + row*(I/2), bank.down_scale + row*(I/16),
                               fp16_to_f32(bank.down_global[row]), g_quant.data() + (size_t) route*I,
                               I, e4m3, g_scale.data() + (size_t) route*(I/16)) * job.weights[route];
                }
                job.y[(size_t) token*H + h] = f32_to_bf16(sum);
            }
        }
    }

    void worker(int tid) {
#if defined(__linux__)
        if (!cores.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cores[tid%cores.size()], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
#else
        (void) tid;
#endif
        uint64_t seen = 0;
        auto last_job = std::chrono::steady_clock::now();
        unsigned polls = 0;
        while (!stop.load(std::memory_order_acquire)) {
            const uint64_t epoch = generation.load(std::memory_order_acquire);
            if (epoch != seen) {
                seen = epoch;
                compute();
                done.fetch_add(1, std::memory_order_acq_rel);
                last_job = std::chrono::steady_clock::now();
                continue;
            }
            pause_worker();
            if ((++polls & 1023u) == 0 && std::chrono::steady_clock::now() - last_job > std::chrono::milliseconds(50)) {
                std::unique_lock<std::mutex> lock(mutex);
                ++sleeping;
                wake.wait(lock, [&] { return stop.load() || generation.load(std::memory_order_acquire) != seen; });
                --sleeping;
            }
        }
    }

    void run(int layer, int tokens, const uint16_t * x, const int32_t * ids, const float * weights,
             uint16_t * y, const int8_t * xq, const float * xs) {
        if (layer < 0 || (size_t) layer >= banks.size() || tokens < 1 || tokens > max_tokens
                || !x || !ids || !weights || !y || (bool(xq) != bool(xs))) {
            throw std::invalid_argument("invalid FreeToken CPU task");
        }
        if (running.test_and_set(std::memory_order_acquire)) {
            throw std::runtime_error("FreeToken CPU executor already has an active task");
        }
        if (skip_empty) {
            bool active = false;
            for (int r = 0; r < tokens*K; ++r) { if (ids[r] >= 0 && ids[r] < E) { active = true; break; } }
            if (!active) {
                std::fill(y,y+size_t(tokens)*H,uint16_t(0));
                running.clear(std::memory_order_release);
                return;
            }
        }
        if (!xq) {
            for (int t = 0; t < tokens; ++t) {
                quantize(x + t*H, H, x_quant.data() + t*H, x_scale.data() + t*(H/16));
            }
            xq = x_quant.data(); xs = x_scale.data();
        }
        job = {layer, tokens, ids, weights, y, xq, xs};
        next_gate.store(0); next_prep.store(0); next_down.store(0); done.store(0);
        {
            std::lock_guard<std::mutex> lock(mutex);
            generation.fetch_add(1, std::memory_order_release);
            if (sleeping) { wake.notify_all(); }
        }
        while (done.load(std::memory_order_acquire) != n_threads) { pause_worker(); }
        running.clear(std::memory_order_release);
    }
};

freetoken_cpu_executor::freetoken_cpu_executor(int h, int i, int e, int k, int tokens, int threads,
        std::vector<freetoken_cpu_bank> banks, std::vector<int> cores) :
    p(new impl(h, i, e, k, tokens, threads, std::move(banks), std::move(cores))) {}
freetoken_cpu_executor::~freetoken_cpu_executor() = default;
const char * freetoken_cpu_executor::isa() const { return p->isa_name; }
void freetoken_cpu_executor::run(int layer, int tokens, const uint16_t * x, const int32_t * ids,
        const float * weights, uint16_t * y, const int8_t * xq, const float * xs) {
    p->run(layer, tokens, x, ids, weights, y, xq, xs);
}
