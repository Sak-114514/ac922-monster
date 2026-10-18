// llama-monpack-loader.cpp — P3 monpack v2 sidecar loader (engine integration)
//
// Contract source of truth: ggml/src/ggml-cuda/monpack/monpack_format.h
// (header / fam descriptors / index records, tile addressing, env switches).
// The CUDA kernel side lives in monpack/monpack-kernels.cu (kernel agent K) and
// exports monpack::register_tensor / is_registered / try_mul_mat / try_mul_mat_id
// via monpack/monpack.h. This loader only ever redirects tensors when those
// kernels are part of the build (CMake defines MONPACK_AVAILABLE).
//
// Loading model:
//   init()                   validate + mmap the sidecar (once, at load_all_data)
//   tensor_src_override(t)   per weight tensor: packed blob pointer or nullptr
//   register_tensor(t)       after the blob bytes reached t's final GPU buffer
//
// The mapping is intentionally never unmapped before process exit.

#include "llama-monpack-loader.h"

#include "llama-impl.h" // LLAMA_LOG_*

#include "ggml.h"
#include "ggml-backend.h"

// format constitution (packed structs, layout_rev=1)
#include "../ggml/src/ggml-cuda/monpack/monpack_format.h"

// NOTE(P3): monpack_format.h's comments say "128B/条目" (fam desc) and "96B 定长"
// (index entry), but the actual `#pragma pack(1)` structs are 88 B and 104 B.
// This loader follows the structs (sizeof-based strides), which is the only
// self-consistent reading of the packed definitions; if the constitution is
// ever amended these asserts force a conscious re-sync of every component.
static_assert(sizeof(monpack_fam_desc) == 88, "monpack fam desc layout drifted from layout_rev=1");
static_assert(sizeof(monpack_index_entry) == 104, "monpack index entry layout drifted from layout_rev=1");

#ifdef MONPACK_AVAILABLE
// kernel contract header (must stay host-safe: it is included from this plain C++ TU)
#include "../ggml/src/ggml-cuda/monpack/monpack.h"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------- hashing ---

struct sha256_ctx {
    uint32_t h[8];
    uint64_t len;
    uint8_t  buf[64];
    size_t   buf_len;
};

constexpr uint32_t MONPACK_K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static inline uint32_t monpack_ror32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_compress(sha256_ctx & c, const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t(p[4 * i + 0]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
               (uint32_t(p[4 * i + 2]) << 8) |  uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = monpack_ror32(w[i - 15], 7) ^ monpack_ror32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = monpack_ror32(w[i - 2], 17) ^ monpack_ror32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = c.h[0], b = c.h[1], g0 = c.h[2], d = c.h[3];
    uint32_t e = c.h[4], f = c.h[5], g1 = c.h[6], h = c.h[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t S1 = monpack_ror32(e, 6) ^ monpack_ror32(e, 11) ^ monpack_ror32(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g1);
        const uint32_t t1 = h + S1 + ch + MONPACK_K256[i] + w[i];
        const uint32_t S0 = monpack_ror32(a, 2) ^ monpack_ror32(a, 13) ^ monpack_ror32(a, 22);
        const uint32_t mj = (a & b) ^ (a & g0) ^ (b & g0);
        const uint32_t t2 = S0 + mj;
        h = g1; g1 = f; f = e; e = d + t1;
        d = g0; g0 = b; b = a; a = t1 + t2;
    }
    c.h[0] += a; c.h[1] += b; c.h[2] += g0; c.h[3] += d;
    c.h[4] += e; c.h[5] += f; c.h[6] += g1; c.h[7] += h;
}

static void sha256_init(sha256_ctx & c) {
    c.h[0] = 0x6a09e667; c.h[1] = 0xbb67ae85; c.h[2] = 0x3c6ef372; c.h[3] = 0xa54ff53a;
    c.h[4] = 0x510e527f; c.h[5] = 0x9b05688c; c.h[6] = 0x1f83d9ab; c.h[7] = 0x5be0cd19;
    c.len = 0;
    c.buf_len = 0;
}

static void sha256_update(sha256_ctx & c, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    c.len += n;
    while (n > 0) {
        const size_t take = std::min(n, sizeof(c.buf) - c.buf_len);
        memcpy(c.buf + c.buf_len, p, take);
        c.buf_len += take;
        p += take;
        n -= take;
        if (c.buf_len == sizeof(c.buf)) {
            sha256_compress(c, c.buf);
            c.buf_len = 0;
        }
    }
}

static void sha256_final(sha256_ctx & c, uint8_t out[32]) {
    const uint64_t bits = c.len * 8; // payload length before padding
    const uint8_t one = 0x80, zero = 0;
    sha256_update(c, &one, 1);
    while (c.buf_len != 56) {
        sha256_update(c, &zero, 1);
    }
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) {
        lenb[i] = (uint8_t)(bits >> (56 - 8 * i));
    }
    sha256_update(c, lenb, sizeof(lenb));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            out[4 * i + j] = (uint8_t)(c.h[i] >> (24 - 8 * j));
        }
    }
}

static uint32_t crc32_ieee(const uint8_t * p, size_t n) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1)));
        }
    }
    return crc ^ 0xffffffffu;
}

// FNV-1a 64 over a NUL-terminated name (index key, monpack_format.h)
static uint64_t fnv1a64(const char * s) {
    uint64_t h = 14695981039346656037ull;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static bool hex_eq_nocase(const char * a, const char * b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const int ca = tolower((unsigned char)a[i]);
        const int cb = tolower((unsigned char)b[i]);
        if (ca != cb) return false;
    }
    return true;
}

// ------------------------------------------------------------------ state ---

struct monpack_state {
    bool initialized = false; // init() ran (idempotency)
    bool active      = false; // sidecar validated and open
    bool verify      = false; // MONSTER_MONPACK_VERIFY=1

    void *       map      = nullptr; // whole-sidecar read-only mapping, held until exit
    size_t       map_size = 0;
    const uint8_t * blob  = nullptr; // map + hdr->blob_off
    uint64_t      blob_total = 0;

    const monpack_header * hdr = nullptr;

    std::unordered_map<uint64_t, const monpack_index_entry *> by_hash;
    std::unordered_set<const ggml_tensor *> pending; // redirected, awaiting register_tensor()

    std::unordered_map<uint32_t, const monpack_fam_desc *> desc_by_type; // fork type -> fam params

    std::unordered_set<uint32_t> type_whitelist; // empty = all packed types
    bool have_type_whitelist = false;

    size_t n_redirect = 0;

    // v2 slice-aware repack stats ([mp-split] summary line)
    size_t   n_meta_tensors = 0;   // tensors handled by tensor_meta_override()
    size_t   n_meta_slices  = 0;   // member slices uploaded + registered
    uint64_t meta_bytes     = 0;   // staging bytes repacked
    size_t   meta_staging_peak = 0;// max concurrent staging bytes (per tensor)
    int64_t  meta_repack_us = 0;   // host repack wall time (excl. H2D)
};

static monpack_state g_state;

static void monpack_reject(const std::string & sidecar, const char * reason) {
    LLAMA_LOG_WARN("monpack: %s: %s — monpack ignored, using original GGUF path\n",
        sidecar.c_str(), reason);
}

// read exactly n bytes at offset from an fd
static bool pread_all(int fd, void * buf, size_t n, uint64_t off) {
    uint8_t * p = (uint8_t *) buf;
    while (n > 0) {
        const ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0) {
            return false;
        }
        p += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return true;
}

// sha256(head 64K || length-encoding || tail 64K) hex, for the encodings a
// converter could plausibly have used for "文件长度" (constitution leaves it open)
static bool gguf_identity_matches(int gguf_fd, uint64_t gguf_size, const char want[64]) {
    const size_t chunk = 64 * 1024;
    const size_t head_n = (size_t)std::min<uint64_t>(chunk, gguf_size);
    const uint64_t tail_off = gguf_size > chunk ? gguf_size - chunk : 0;
    const size_t tail_n = (size_t)(gguf_size - tail_off);

    std::vector<uint8_t> head(head_n), tail(tail_n);
    if (!pread_all(gguf_fd, head.data(), head_n, 0)) return false;
    if (!pread_all(gguf_fd, tail.data(), tail_n, tail_off)) return false;

    const auto try_encoding = [&](const void * lenb, size_t lenb_n) -> bool {
        sha256_ctx c;
        sha256_init(c);
        sha256_update(c, head.data(), head_n);
        sha256_update(c, lenb, lenb_n);
        sha256_update(c, tail.data(), tail_n);
        uint8_t d[32];
        sha256_final(c, d);
        static const char hx[] = "0123456789abcdef";
        char hex[65];
        for (int i = 0; i < 32; i++) {
            hex[2 * i + 0] = hx[d[i] >> 4];
            hex[2 * i + 1] = hx[d[i] & 0xf];
        }
        hex[64] = '\0';
        return hex_eq_nocase(hex, want, 64);
    };

    { // 8-byte little-endian u64
        uint8_t lenb[8];
        for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)(gguf_size >> (8 * i));
        if (try_encoding(lenb, sizeof(lenb))) return true;
    }
    { // 8-byte big-endian u64
        uint8_t lenb[8];
        for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)(gguf_size >> (56 - 8 * i));
        if (try_encoding(lenb, sizeof(lenb))) return true;
    }
    { // ASCII decimal
        char lenb[32];
        const int n = snprintf(lenb, sizeof(lenb), "%llu", (unsigned long long) gguf_size);
        if (try_encoding(lenb, (size_t) n)) return true;
    }
    return false;
}

} // namespace

// ---------------------------------------------------------------- public ---

void llama_monpack::init(const char * gguf_path) {
    monpack_state & S = g_state;
    if (S.initialized) {
        return;
    }
    S.initialized = true;

    if (gguf_path == nullptr || gguf_path[0] == '\0') {
        return; // no primary file (from-buffer load) — nothing to bind to
    }
    const std::string sidecar = std::string(gguf_path) + ".monpack";

    const char * env = getenv("MONSTER_MONPACK");
    if (env != nullptr && atoi(env) == 0) {
        LLAMA_LOG_INFO("monpack: disabled via MONSTER_MONPACK=0, using original GGUF path\n");
        return;
    }

#ifndef MONPACK_AVAILABLE
    // No monpack CUDA kernels in this build: redirecting weights would hand the stock
    // kernels packed bytes they cannot read, so stay entirely on the original path.
    if (access(sidecar.c_str(), F_OK) == 0) {
        LLAMA_LOG_INFO("monpack: sidecar '%s' found but this build has no monpack kernels, "
                       "using original GGUF path\n", sidecar.c_str());
    }
    return;
#else
    if (access(sidecar.c_str(), F_OK) != 0) {
        return; // no sidecar — normal, silent
    }

    // optional per-tensor blob hash verification (MONSTER_MONPACK_VERIFY=1)
    if (const char * venv = getenv("MONSTER_MONPACK_VERIFY")) {
        S.verify = atoi(venv) != 0;
    }

    // fork-type whitelist (MONSTER_MONPACK_TYPES="20,21,..."; empty/unset = all)
    if (const char * tenv = getenv("MONSTER_MONPACK_TYPES")) {
        const std::string s(tenv);
        size_t pos = 0;
        while (pos <= s.size()) {
            const size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            while (!tok.empty() && tok.back() == ' ') tok.pop_back();
            while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
            if (!tok.empty()) {
                const long v = strtol(tok.c_str(), nullptr, 10);
                if (v > 0 && v <= 0x7fffffff) {
                    S.type_whitelist.insert((uint32_t) v);
                }
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    S.have_type_whitelist = !S.type_whitelist.empty();

    // ---- open + map the sidecar (kept until process exit) ----
    const int fd = open(sidecar.c_str(), O_RDONLY);
    if (fd < 0) {
        monpack_reject(sidecar, "open failed");
        return;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t) 4096) {
        close(fd);
        monpack_reject(sidecar, "stat failed or file too small");
        return;
    }
    const size_t size = (size_t) st.st_size;
    void * map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        monpack_reject(sidecar, "mmap failed");
        return;
    }

    const auto reject = [&](const char * reason) {
        // keep the mapping (never munmap mid-run, per spec); just deactivate
        S.map = map;
        S.map_size = size;
        monpack_reject(sidecar, reason);
    };

    // ---- header ----
    const monpack_header * hdr = (const monpack_header *) map;
    if (memcmp(hdr->magic, MONPACK_MAGIC, 4) != 0) {
        reject("bad magic");
        return;
    }
    if (hdr->version != MONPACK_VERSION || hdr->layout_rev != MONPACK_LAYOUT_REV) {
        reject("unsupported version/layout_rev");
        return;
    }
    if ((hdr->flags & 0x1) == 0) {
        reject("sidecar data is not little-endian");
        return;
    }

    // ---- region bounds ----
    const size_t HDR_SIZE = 4096;
    if (hdr->desc_off < HDR_SIZE || hdr->const_off < hdr->desc_off ||
            hdr->blob_off < hdr->const_off || hdr->index_off < hdr->blob_off ||
            (uint64_t) size < hdr->index_off) {
        reject("region offsets out of order or beyond file size");
        return;
    }
    const uint64_t desc_bytes = (uint64_t) hdr->n_fams * sizeof(monpack_fam_desc);
    const uint64_t index_bytes = (uint64_t) hdr->n_tensors * sizeof(monpack_index_entry);
    if (hdr->desc_off + desc_bytes > hdr->const_off ||
            hdr->blob_off + hdr->blob_total > hdr->index_off ||
            hdr->index_off + index_bytes > (uint64_t) size) {
        reject("descriptor/index/blob region exceeds file");
        return;
    }

    // ---- gguf identity binding: sha256(head 64K || len || tail 64K) ----
    {
        const int gfd = open(gguf_path, O_RDONLY);
        if (gfd < 0) {
            reject("cannot open gguf for identity check");
            return;
        }
        struct stat gst;
        if (fstat(gfd, &gst) != 0) {
            close(gfd);
            reject("cannot stat gguf for identity check");
            return;
        }
        const bool ok = gguf_identity_matches(gfd, (uint64_t) gst.st_size, hdr->gguf_identity);
        close(gfd);
        if (!ok) {
            reject("gguf_identity mismatch (sidecar built for a different gguf)");
            return;
        }
    }

    // ---- descriptor table: fork type -> family params ----
    std::unordered_map<uint32_t, const monpack_fam_desc *> desc_by_type;
    for (uint32_t i = 0; i < hdr->n_fams; i++) {
        const monpack_fam_desc * d = (const monpack_fam_desc *)((const uint8_t *) map + hdr->desc_off + (uint64_t) i * sizeof(monpack_fam_desc));
        if (d->family_id > MONPACK_FAM_C) {
            reject("descriptor with unknown family_id");
            return;
        }
        if (d->family_id != MONPACK_FAM_PASS && d->code_bytes + 2 != d->block_bytes) {
            reject("descriptor code_bytes inconsistent with block_bytes");
            return;
        }
        desc_by_type[d->ggml_type] = d;
    }
    S.desc_by_type = desc_by_type; // v2: slice repack needs block_size / code_bytes

    // ---- index ----
    const uint8_t * index_base = (const uint8_t *) map + hdr->index_off;
    if (hdr->index_crc32 != 0 && crc32_ieee(index_base, (size_t) index_bytes) != hdr->index_crc32) {
        reject("index crc32 mismatch");
        return;
    }

    uint64_t blob_sum = 0;
    size_t n_packed = 0;
    for (uint32_t i = 0; i < hdr->n_tensors; i++) {
        const monpack_index_entry * e = (const monpack_index_entry *)(index_base + (uint64_t) i * sizeof(monpack_index_entry));

        if (e->family_id == MONPACK_FAM_PASS) {
            if (e->blob_len != 0) {
                reject("pass-through entry with nonzero blob");
                return;
            }
            continue;
        }

        // Passthrough entries (family 0) stay on the original GGUF path and have no
        // descriptor; only packed entries must resolve to a matching descriptor.
        const monpack_fam_desc * d = nullptr;
        if (e->family_id != 0) {
            const auto dit = desc_by_type.find(e->ggml_type);
            if (dit == desc_by_type.end() || dit->second->family_id != e->family_id) {
                reject("entry without matching type descriptor");
                return;
            }
            d = dit->second;
        }

        if (e->family_id != 0) {
            // tile geometry cross-check (layout_rev=1 addressing, monpack_format.h)
            if (e->k_tile == 0 || e->k_tile % d->block_size != 0 || e->ne_k % e->k_tile != 0 ||
                    e->n_tiles_n != (e->ne_n + 63) / 64 || e->n_tiles_k != e->ne_k / e->k_tile ||
                    e->ne_e == 0) {
                reject("entry tile geometry inconsistent");
                return;
            }
            const uint64_t nb = e->k_tile / d->block_size;
            const uint64_t tile_bytes = nb * 64 * (2 + d->code_bytes);
            if (e->blob_len != (uint64_t) e->ne_e * e->n_tiles_n * e->n_tiles_k * tile_bytes) {
                reject("entry blob_len violates invariant I1/tile layout");
                return;
            }
            // entry blob_off is a file-absolute offset; the blob region ends at
            // hdr->blob_off + hdr->blob_total (blob_total excludes the base).
            if (e->blob_off < hdr->blob_off || e->blob_off + e->blob_len > hdr->blob_off + hdr->blob_total) {
                reject("entry blob out of blob region");
                return;
            }
        } else if (e->blob_len != 0) {
            reject("passthrough entry with nonzero blob_len");
            return;
        }

        blob_sum += e->blob_len;
        if (S.have_type_whitelist && S.type_whitelist.count(e->ggml_type) == 0) {
            continue; // gray-release: this fork type keeps the original path
        }
        n_packed++;
        S.by_hash.emplace(e->name_hash, e);
    }
    if (blob_sum != hdr->blob_total) {
        reject("blob_total does not match sum of entry blob_len");
        return;
    }

    S.map = map;
    S.map_size = size;
    S.hdr = hdr;
    S.blob = (const uint8_t *) map; // entries carry file-absolute blob_off (K2 fix: no double base)
    S.blob_total = hdr->blob_total;
    S.active = true;

    LLAMA_LOG_INFO("monpack: sidecar '%s' active: %u index entries, %zu packed tensors in this run, "
                   "blob %.2f GiB, fork_tag '%.32s'\n",
        sidecar.c_str(), hdr->n_tensors, n_packed, (double) hdr->blob_total / (1024.0 * 1024.0 * 1024.0),
        hdr->fork_tag);
#endif // MONPACK_AVAILABLE
}

const void * llama_monpack::tensor_src_override(ggml_tensor * t) {
    monpack_state & S = g_state;
    if (!S.active || t == nullptr) {
        return nullptr;
    }

    // Only weights that live in a GPU device buffer may be redirected: CPU-resident
    // tensors keep the plain GGUF layout so every non-CUDA consumer stays correct.
    if (t->data == nullptr || t->buffer == nullptr) {
        return nullptr;
    }
    {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev == nullptr) {
            return nullptr;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            return nullptr;
        }
    }

    const char * name = ggml_get_name(t);
    if (name == nullptr || name[0] == '\0') {
        return nullptr;
    }
    const auto it = S.by_hash.find(fnv1a64(name));
    if (it == S.by_hash.end() || it->second->family_id == MONPACK_FAM_PASS) {
        return nullptr;
    }
    const monpack_index_entry * e = it->second;

    // defensive model-vs-index consistency (identity already bound the files)
    if (e->ggml_type != (uint32_t) t->type ||
            e->ne_k != (uint32_t) t->ne[0] || e->ne_n != (uint32_t) t->ne[1] ||
            (e->ne_e > 1 && e->ne_e != (uint32_t) t->ne[2]) ||
            e->src_len != ggml_nbytes(t)) {
        static bool warned = false; // a mismatch means a corrupt binding; stay safe
        if (!warned) {
            LLAMA_LOG_WARN("monpack: index entry for '%s' does not match the model tensor — "
                           "skipping monpack for it (and future mismatches)\n", name);
            warned = true;
        }
        return nullptr;
    }

    if (S.verify) {
        // truncated-sha256 self-check of the packed blob (MONSTER_MONPACK_VERIFY=1)
        uint8_t d[32];
        {
            sha256_ctx c;
            sha256_init(c);
            sha256_update(c, S.blob + e->blob_off, (size_t) e->blob_len);
            sha256_final(c, d);
        }
        if (memcmp(d, e->blob_sha128, sizeof(e->blob_sha128)) != 0) {
            LLAMA_LOG_WARN("monpack: blob hash mismatch for '%s' — skipping monpack for it\n", name);
            return nullptr;
        }
    }

    S.pending.insert(t);
    S.n_redirect++;
    return S.blob + e->blob_off;
}

void llama_monpack::register_tensor(ggml_tensor * t) {
    monpack_state & S = g_state;
    if (!S.active || t == nullptr) {
        return;
    }
    if (S.pending.erase(t) == 0) {
        return; // was not redirected through tensor_src_override()
    }

#ifdef MONPACK_AVAILABLE
    // t->data now holds the packed blob in its final (device) buffer; the monpack
    // registry derives family / k-tile / n-tile geometry from t->type and t->ne
    // per monpack_format.h (k-tile rule + fixed N-tile 64).
    monpack::register_tensor(t);
#endif
}

// ---------------------------------------------------------------------------
// v2: slice-aware repack for island (meta buffer) tensors
//
// The island TP splits expert tensors along K (ne0). The meta backend materializes
// one member tensor per device with ne = [K_j, N, E] (original-layout K interval
// [k0_j, k0_j + K_j)), and the stock load path scatters the GGUF bytes into them.
// The sidecar blob is packed with FULL-tensor tile addressing, so those bytes can
// not be scattered byte-for-byte: instead, for every member slice we synthesize a
// standalone packed buffer that follows the very same constitution formula but
// with the slice geometry (K_j, N, E) and a re-derived k_tile (K%256->256, %128
// ->128, %64->64). Packed bytes == original slice bytes for every geometry whose
// N tiles by 64 (I1 at slice granularity), so the staging buffer overrides the
// member upload in place. Every block's d/code bytes are gathered from the full
// blob with full-tensor addressing — a pure byte permutation, no dequant.
// ---------------------------------------------------------------------------

#ifdef MONPACK_AVAILABLE

namespace {

struct slice_repack_ctx {
    const uint8_t * blob;  // full-tensor packed blob (inside the sidecar mmap)
    uint64_t K, N, E;      // full logical geometry
    uint32_t ktf;          // full-tensor k_tile (index entry)
    uint32_t bs, cb;       // block_size, code_bytes (fam desc)
    uint32_t nbf;          // ktf / bs
    uint64_t ntkf;         // K / ktf
    uint64_t ntn;          // N / 64
    uint64_t tile_bytes_f; // nbf * 64 * (2 + cb)
};

// slice k_tile per constitution rule; caller has already validated Ks % 64 == 0
static inline uint32_t slice_k_tile(uint64_t Ks) {
    if (Ks % 256 == 0) return 256;
    if (Ks % 128 == 0) return 128;
    return 64;
}

// One expert's slice rows: byte gather per (row, block-run). Both ends of a run
// stay inside one tile on source and destination, so each run is one memcpy for
// the d domain and one for the code domain. Covers both split flavors:
//   K-split (axis 0): k0/Ks carve the k interval, row_off = 0;
//   N-split (axis 1): row_off/Ns carve the row (n) interval, k0 = 0, Ks = K.
// Source rows/tiles are addressed with the GLOBAL row (r + row_off).
static void slice_repack_expert(const slice_repack_ctx & R, uint32_t k0, uint32_t Ks,
        uint32_t row_off, uint32_t Ns, uint32_t kts, uint8_t * dst, uint64_t e) {
    const uint32_t nbs  = kts / R.bs;
    const uint64_t ntks = Ks / kts;
    const uint64_t tbs  = (uint64_t) nbs * 64 * (2 + R.cb);
    const uint64_t kb_blocks = Ks / R.bs;
    const uint64_t kbg0 = k0 / R.bs; // first global block index of the slice

    for (uint64_t r = 0; r < Ns; r++) {
        const uint64_t rg  = r + row_off;  // global row in the full tensor
        const uint64_t n0  = r / 64;       // destination n-tile (local)
        const uint64_t rin = r % 64;       // destination row-in-tile (local)
        const uint64_t gr  = rg % 64;      // source row-in-tile (global)
        const uint64_t ntn_s = Ns / 64;    // slice n-tile count (dst layout)
        uint8_t * dst_tile_row = dst + ((e * ntn_s + n0) * ntks) * tbs;
        const uint8_t * src_row_tile = R.blob + ((e * R.ntn + rg / 64) * R.ntkf) * R.tile_bytes_f;
        for (uint64_t kb = 0; kb < kb_blocks;) {
            const uint64_t g0 = kbg0 + kb;
            const uint64_t run = std::min(std::min<uint64_t>(nbs - kb % nbs, R.nbf - g0 % R.nbf),
                                          kb_blocks - kb);
            const uint64_t d_in  = (rin * nbs + kb % nbs) * 2;
            const uint64_t c_in  = (rin * nbs + kb % nbs) * R.cb;
            const uint64_t d_inF = (gr * R.nbf + g0 % R.nbf) * 2;
            const uint64_t c_inF = (gr * R.nbf + g0 % R.nbf) * R.cb;

            const uint8_t * src_tile = src_row_tile + (g0 / R.nbf) * R.tile_bytes_f;
            uint8_t * dst_tile = dst_tile_row + (kb / nbs) * tbs;

            memcpy(dst_tile + d_in, src_tile + d_inF, run * 2);
            memcpy(dst_tile + nbs * 64 * 2 + c_in,
                   src_tile + R.nbf * 64 * 2 + c_inF, run * R.cb);
            kb += run;
        }
    }
}

} // namespace

#endif // MONPACK_AVAILABLE

bool llama_monpack::tensor_meta_override(ggml_tensor * t) {
    monpack_state & S = g_state;
    if (!S.active || t == nullptr || t->buffer == nullptr) {
        return false;
    }
#ifndef MONPACK_AVAILABLE
    return false; // S.active is only ever true with kernels in the build; keep static checkers happy
#else
    if (!ggml_backend_buffer_is_meta(t->buffer)) {
        return false; // plain (non-island) buffers keep the tensor_src_override() path
    }

    const char * name = ggml_get_name(t);
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const auto it = S.by_hash.find(fnv1a64(name));
    if (it == S.by_hash.end() || it->second->family_id == MONPACK_FAM_PASS) {
        return false; // not packed (or gray-released type): original path
    }
    const monpack_index_entry * e = it->second;

    const auto dit = S.desc_by_type.find(e->ggml_type);
    if (dit == S.desc_by_type.end() || dit->second->family_id != e->family_id) {
        return false; // cannot happen (validated in init); stay safe
    }
    const monpack_fam_desc * d = dit->second;

    // full model tensor must match the index entry exactly
    if (e->ggml_type != (uint32_t) t->type ||
            e->ne_k != (uint32_t) t->ne[0] || e->ne_n != (uint32_t) t->ne[1] ||
            (e->ne_e > 1 && e->ne_e != (uint32_t) t->ne[2]) ||
            e->src_len != ggml_nbytes(t) ||
            (uint64_t) ggml_blck_size(t->type) != d->block_size) {
        LLAMA_LOG_WARN("monpack: [mp-split] '%s': index entry does not match the model tensor — "
                       "original path for it\n", name);
        return false;
    }

    if (S.verify) {
        uint8_t sha[32];
        sha256_ctx c;
        sha256_init(c);
        sha256_update(c, S.blob + e->blob_off, (size_t) e->blob_len);
        sha256_final(c, sha);
        if (memcmp(sha, e->blob_sha128, sizeof(e->blob_sha128)) != 0) {
            LLAMA_LOG_WARN("monpack: blob hash mismatch for '%s' — original path for it\n", name);
            return false;
        }
    }

    // ---- split geometry: only plain single-segment K-split (axis 0) or N-split
    //      (axis 1) can be repacked; segments (fused QKV-style) fall back ----
    const ggml_backend_meta_split_state ss = ggml_backend_meta_tensor_get_split_state(t);
    const bool axis_k = ss.axis == GGML_BACKEND_SPLIT_AXIS_0;
    const bool axis_n = ss.axis == GGML_BACKEND_SPLIT_AXIS_1;
    if ((!axis_k && !axis_n) || ss.n_segments != 1 || ss.nr[0] != 1) {
        LLAMA_LOG_WARN("monpack: [mp-split] '%s': island split is axis=%d segs=%u nr=%u "
                       "(not plain K/N) — original path for it\n",
            name, (int) ss.axis, ss.n_segments, ss.nr[0]);
        return false;
    }
    const size_t n_members = ggml_backend_meta_buffer_get_n_simple(t->buffer);
    if (n_members == 0 || n_members > GGML_BACKEND_META_MAX_DEVICES) {
        LLAMA_LOG_WARN("monpack: [mp-split] '%s': bad island member count %zu — original path\n",
            name, n_members);
        return false;
    }
    // N must tile by 64 exactly, otherwise packed slice bytes != original slice
    // bytes and the member buffer could not hold the staging layout in place.
    if (t->ne[1] % 64 != 0 || (uint64_t) e->n_tiles_n * 64 != (uint64_t) t->ne[1]) {
        LLAMA_LOG_WARN("monpack: [mp-split] '%s': N=%lld not a multiple of 64 — original path\n",
            name, (long long) t->ne[1]);
        return false;
    }
    // the grouped MoE dispatch path only covers ne_e <= 512 (monpack.h); a larger
    // expert count would leave registered-but-never-dispatched packed bytes
    if (e->ne_e > 512) {
        LLAMA_LOG_WARN("monpack: [mp-split] '%s': ne_e=%u > 512 not covered by the dispatch "
                       "hooks — original path\n", name, e->ne_e);
        return false;
    }
    // K-split: k_tile is re-derived per member slice; N-split: K is unsplit, so the
    // slice k_tile follows the same rule on the full K.
    const uint32_t kts_axis_n = axis_n ? slice_k_tile((uint64_t) t->ne[0]) : 0;
    if (axis_n && ((uint64_t) t->ne[0] % kts_axis_n != 0)) {
        LLAMA_LOG_WARN("monpack: [mp-split] '%s': K=%lld not tileable for N-split — original path\n",
            name, (long long) t->ne[0]);
        return false;
    }

    // ---- validate every member slice BEFORE touching any bytes (all-or-nothing) ----
    struct member_slice {
        uint32_t k0, ks;   // k interval of this member (axis 0)
        uint32_t n0, ns;   // row interval of this member (axis 1)
        ggml_tensor * st;
        uint32_t kts;      // slice k_tile
    };
    std::vector<member_slice> slices;
    slices.reserve(n_members);
    uint64_t share_sum = 0;
    for (size_t j = 0; j < n_members; j++) {
        const int64_t share = ss.ne[j];
        if (share < 0) {
            LLAMA_LOG_WARN("monpack: [mp-split] '%s': negative member share — original path\n", name);
            return false;
        }
        if (share == 0) {
            continue; // this member holds nothing of this tensor
        }
        ggml_tensor * st = ggml_backend_meta_tensor_get_simple(t, j);
        if (st == nullptr) {
            LLAMA_LOG_WARN("monpack: [mp-split] '%s': member %zu not materialized — original path\n",
                name, j);
            return false;
        }
        uint32_t k0 = 0, ks, n0 = 0, ns, kts;
        if (axis_k) {
            ks = (uint32_t) share; k0 = (uint32_t) share_sum; ns = (uint32_t) t->ne[1];
            if ((uint64_t) ks % d->block_size != 0 || (uint64_t) k0 % d->block_size != 0) {
                LLAMA_LOG_WARN("monpack: [mp-split] '%s': member %zu K=[%u,%u) not block-aligned "
                               "(bs=%u) — original path\n", name, j, k0, k0 + ks, d->block_size);
                return false;
            }
            kts = slice_k_tile(ks);
            if (ks % kts != 0) {
                LLAMA_LOG_WARN("monpack: [mp-split] '%s': member %zu K=%u not tileable — original path\n",
                    name, j, ks);
                return false;
            }
        } else {
            ns = (uint32_t) share; n0 = (uint32_t) share_sum; ks = (uint32_t) t->ne[0];
            kts = kts_axis_n;
            if (ns % 64 != 0 || n0 % 64 != 0) {
                LLAMA_LOG_WARN("monpack: [mp-split] '%s': member %zu N=[%u,%u) not row-tile aligned "
                               "— original path\n", name, j, n0, n0 + ns);
                return false;
            }
        }
        if (st->type != t->type || (uint64_t) st->ne[0] != (uint64_t) ks ||
                (uint64_t) st->ne[1] != (uint64_t) ns || st->ne[2] != t->ne[2]) {
            LLAMA_LOG_WARN("monpack: [mp-split] '%s': member %zu geometry [%lld,%lld,%lld] != share "
                           "[%u,%u,%lld] — original path\n", name, j,
                (long long) st->ne[0], (long long) st->ne[1], (long long) st->ne[2],
                ks, ns, (long long) t->ne[2]);
            return false;
        }
        slices.push_back({ k0, ks, n0, ns, st, kts });
        share_sum += (uint64_t) share;
    }
    const uint64_t axis_len = axis_k ? (uint64_t) t->ne[0] : (uint64_t) t->ne[1];
    if (share_sum != axis_len) {
        LLAMA_LOG_WARN("monpack: [mp-split] '%s': member shares sum to %llu, axis len=%llu — "
                       "original path\n", name, (unsigned long long) share_sum,
            (unsigned long long) axis_len);
        return false;
    }
    if (slices.empty()) {
        return false; // nothing resident on any member: nothing to do either way
    }

    // ---- allocate all staging up front (a later malloc failure must not leave
    //      the tensor half packed) ----
    const size_t n_stg = slices.size();
    std::vector<uint8_t *> stg(n_stg, nullptr);
    size_t stg_total = 0;
    for (size_t i = 0; i < n_stg; i++) {
        stg[i] = (uint8_t *) malloc(ggml_nbytes(slices[i].st));
        if (stg[i] == nullptr) {
            for (size_t k = 0; k < i; k++) free(stg[k]);
            LLAMA_LOG_WARN("monpack: [mp-split] '%s': staging malloc failed (%zu bytes) — "
                           "original path\n", name, stg_total);
            return false;
        }
        stg_total += ggml_nbytes(slices[i].st);
    }

    // ---- host repack (parallel over experts; pure byte gather) ----
    slice_repack_ctx R;
    R.blob  = S.blob + e->blob_off;
    R.K     = e->ne_k;
    R.N     = e->ne_n;
    R.E     = e->ne_e;
    R.ktf   = e->k_tile;
    R.bs    = d->block_size;
    R.cb    = d->code_bytes;
    R.nbf   = R.ktf / R.bs;
    R.ntkf  = R.K / R.ktf;
    R.ntn   = R.N / 64;
    R.tile_bytes_f = (uint64_t) R.nbf * 64 * (2 + R.cb);

    const auto repack_begin = std::chrono::steady_clock::now();
    const unsigned n_threads = (unsigned) std::min<uint64_t>(32, R.E);
    if (n_threads <= 1) {
        for (uint64_t eidx = 0; eidx < R.E; eidx++) {
            for (size_t i = 0; i < n_stg; i++) {
                slice_repack_expert(R, slices[i].k0, slices[i].ks, slices[i].n0, slices[i].ns, slices[i].kts, stg[i], eidx);
            }
        }
    } else {
        std::atomic<uint64_t> next_e { 0 };
        auto worker = [&]() {
            uint64_t eidx;
            while ((eidx = next_e.fetch_add(1)) < R.E) {
                for (size_t i = 0; i < n_stg; i++) {
                    slice_repack_expert(R, slices[i].k0, slices[i].ks, slices[i].n0, slices[i].ns, slices[i].kts, stg[i], eidx);
                }
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(n_threads - 1);
        for (unsigned th = 1; th < n_threads; th++) {
            pool.emplace_back(worker);
        }
        worker();
        for (auto & th : pool) {
            th.join();
        }
    }
    const auto repack_end = std::chrono::steady_clock::now();
    const int64_t repack_us = std::chrono::duration_cast<std::chrono::microseconds>(repack_end - repack_begin).count();

    // ---- register every member FIRST and verify, before any byte is uploaded ----
    // The host-side checks above mirror the kernel registry's acceptance rules
    // (monpack.h): tile-rule k_tile, packed type whitelist, family table, ne_e<=512.
    // If the registry still rejects a member that is a contract break — abort the
    // load loudly instead of risking registered-pointers-with-original-bytes.
    for (size_t i = 0; i < n_stg; i++) {
        monpack::register_tensor(slices[i].st);
        if (!monpack::is_registered(slices[i].st->data)) {
            GGML_ABORT("monpack: [mp-split] '%s': member %zu (K=%u) rejected by the kernel "
                       "registry despite passing host-side validation", name, i, slices[i].ks);
        }
    }

    // ---- upload + bookkeeping member by member ----
    std::string shares;
    for (size_t i = 0; i < n_stg; i++) {
        const size_t nb = ggml_nbytes(slices[i].st);
        ggml_backend_tensor_set(slices[i].st, stg[i], 0, nb);
        free(stg[i]);
        stg[i] = nullptr;
        shares += (i ? "," : "") + std::to_string(slices[i].ks) + "/" + std::to_string(slices[i].kts);
    }

    S.n_meta_tensors++;
    S.n_meta_slices += n_stg;
    S.meta_bytes    += stg_total;
    S.meta_staging_peak = std::max(S.meta_staging_peak, stg_total);
    S.meta_repack_us += repack_us;

    LLAMA_LOG_INFO("monpack: [mp-split] %s: K=%u k_tile=%u -> members(K/k_tile)=[%s], "
                   "staging %.2f MiB, repack %.1f ms\n",
        name, e->ne_k, e->k_tile, shares.c_str(),
        (double) stg_total / (1024.0 * 1024.0), (double) repack_us / 1000.0);
    if (S.n_meta_tensors % 32 == 0) {
        LLAMA_LOG_INFO("monpack: [mp-split] running total: %zu tensors, %zu slices, %.2f GiB repacked, "
                       "staging peak %.2f MiB, repack %.2f s\n",
            S.n_meta_tensors, S.n_meta_slices, (double) S.meta_bytes / (1024.0 * 1024.0 * 1024.0),
            (double) S.meta_staging_peak / (1024.0 * 1024.0), (double) S.meta_repack_us / 1e6);
    }
    return true;
#endif // MONPACK_AVAILABLE
}
