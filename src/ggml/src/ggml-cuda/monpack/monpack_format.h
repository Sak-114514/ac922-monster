// monpack_format.h — P3 monpack v2 边车格式宪法（layout_rev=1）
// 三个组件（python 转换器 / CUDA 内核 / 加载器）共同遵守本文件，任何偏差以本文件为准。
// 布局核心不变量：
//   I1. 纯字节重排：每张量 blob 字节数 == 原 GGUF 张量字节数（d 域分离不改总量）
//   I2. k 序不变：每行的块在 k 方向保持原顺序（GEMV 逐位等价的前提）
//   I3. 专家维最外：expert e 的全部 tile 连续，grouped 内核按专家取 tile 不跨段
//   I4. 块内字节序不变：块内部（除 d 的 2 字节被抽走）保持原始字节顺序
#pragma once
#include <stdint.h>

#define MONPACK_MAGIC       "MPK2"
#define MONPACK_VERSION     2u
#define MONPACK_LAYOUT_REV  1u
#define MONPACK_TILE_N      64    // N-tile 固定 64 行
// K-tile 规则（按整除性自动降级；block_size 必须 | k_tile）：
//   k_tile = (K%256==0) ? 256 : (K%128==0) ? 128 : (K%64==0) ? 64 : 0(拒绝打包)
//   每张量 blob_off 按 256B 对齐（内核向量加载友好）；index_crc32 = zlib.crc32(整个 index 区字节)
//   本检查点：gate/up K=2560 → 256；down K=640 → 128；FAM-B(bs=256) 全在 K=2560 侧 ✓

// ---- 族划分 ----
#define MONPACK_FAM_PASS 0   // 透传（PLE / BF16 / K-quant 稠密），不打包
#define MONPACK_FAM_A    1   // IQ4_NL：16 项码本 + d
#define MONPACK_FAM_B    2   // IQ2_XXS/XS/S、IQ3_XXS/S：grid + ksigns + 二级 scale
#define MONPACK_FAM_C    3   // Q8_0、Q2_0(fork, QK=64)：均匀/简单反量化

#pragma pack(push, 1)

// ---- 头部（文件头 4KB 内）----
struct monpack_header {
    char     magic[8];          // "MPK2"
    uint32_t version;           // 2
    uint32_t layout_rev;        // 1
    uint32_t flags;             // bit0=小端数据；其余 0
    char     gguf_identity[64]; // sha256_hex(gguf 前64KB ‖ u64LE(文件长度8B) ‖ 后64KB) —— 边车↔GGUF 绑定（长度编码钉死 u64 小端）
    char     fork_tag[32];      // "monster-gsq-rco"
    uint32_t n_tensors;         // index 条目数（含透传记录）
    uint32_t n_fams;            // 描述表条目数
    uint64_t desc_off;          // 类型描述表偏移（紧跟头 4KB）
    uint64_t const_off;         // 常量区偏移（各类型码本/grid 原始字节，供校验）
    uint64_t blob_off;          // blob 区偏移
    uint64_t index_off;         // index 区偏移（文件尾）
    uint64_t blob_total;        // blob 总字节
    uint32_t index_crc32;
    uint32_t reserved;
};

// ---- 类型描述（88B/条目，#pragma pack(1)，以 sizeof(struct) 为唯一权威）----
struct monpack_fam_desc {
    uint32_t family_id;         // 1/2/3
    uint32_t ggml_type;         // fork GGML_TYPE_* 值
    uint32_t block_size;        // 每块权重数（32/64/256）
    uint32_t block_bytes;       // GGUF 每块字节数（含 2B d）
    uint32_t code_bytes;        // block_bytes - 2
    uint32_t scale_levels;      // FAM-B=16，其余 0
    uint32_t sign_src;          // 0=无 1=ksigns_iq2xs
    uint32_t grid_entries;      // LUT 项数，0=无
    uint32_t grid_dim;          // 每 grid 项覆盖权重数（iq2xxs=8, iq3xxs=4, iq4nl=1）
    uint32_t reserved0[2];
    uint32_t lut_mask;          // bit0 kvalues_iq4nl | bit1 iq2xxs_grid | bit2 iq2xs_grid
                                // | bit3 iq2s_grid | bit4 iq3xxs_grid | bit5 iq3s_grid
                                // | bit6 ksigns_iq2xs | bit7 ksigns_iq2ns | bit8 kmask_iq2xs
    uint32_t dequant_rev;       // 反量化代数版本（T1 用）
    char     code_name[16];     // "IQ4_NL" 等
    uint8_t  reserved[20];
};

// ---- index 记录（104B 定长，文件尾，以 sizeof(struct) 为唯一权威）----
struct monpack_index_entry {
    uint64_t name_hash;         // FNV-1a 64 of tensor name
    uint32_t gguf_tensor_idx;
    uint32_t ggml_type;         // fork 类型号
    uint32_t family_id;         // 0=透传（blob_len=0，引擎走 GGUF 原路径）
    uint32_t ne_k, ne_n, ne_e;  // 逻辑形状（K 为连续维）
    uint32_t k_tile;            // 本张量 K-tile（256/128/64）
    uint32_t n_tiles_n;         // ne_n / 64（ne_n 非 64 倍数时 ceil，补行 d=0 全零）
    uint32_t n_tiles_k;         // ne_k / k_tile
    uint64_t blob_off, blob_len;
    uint64_t src_len;           // 原 GGUF 张量字节数（== blob_len，I1）
    uint8_t  src_sha128[16];    // 原 GGUF 张量字节截断 sha256（T0 源校验）
    uint8_t  blob_sha128[16];   // 重排 blob 截断 sha256
    uint32_t reserved;
};
#pragma pack(pop)

// ---- tile 布局（layout_rev=1 核心，唯一权威定义）----
// 张量按 (e, n0, k0) 三层循环写入 blob：e 最外，n0 步 64，k0 步 k_tile 最内。
// tile 内部：
//   nb = k_tile / block_size                       （每行每 tile 的块数）
//   d 域   : nb * 64 * 2 字节，序 = (row, kb) 行主序，每项 = 该块原始的 2B ggml_half d
//   code 域: nb * 64 * code_bytes 字节，序 = (row, kb) 行主序，
//            每项 = 该块去掉头 2B d 后的剩余字节（原始顺序，I4）
// 寻址（设备/宿主通用公式）：
//   tile_off(e, n0, k0) = blob_off + ((e * n_tiles_n + n0/64) * n_tiles_k + k0/k_tile) * tile_bytes
//   tile_bytes = nb * 64 * (2 + code_bytes)
//   块 (row, kb) 的 d 地址   = tile_off + (row * nb + kb) * 2
//   块 (row, kb) 的 code 地址 = tile_off + nb*64*2 + (row * nb + kb) * code_bytes
//   该块覆盖 k 区间 [k0 + kb*block_size, k0 + (kb+1)*block_size)
// 补行（ne_n % 64 != 0 时）：d=0 → 反量化为全零，无需特判。本检查点无此情形。

// ---- 目标检查点的类型参数表（ggml-common.h 钉死，bpw 全部对账吻合）----
// ggml_type | family | block_size | block_bytes | 说明
//    20     |   A    |     32     |     18      | IQ4_NL:  d + 16B nibble 码（kvalues_iq4nl[16]）
//    8      |   C    |     32     |     34      | Q8_0:   d + 32B int8
//    42     |   C    |     64     |     18      | Q2_0(fork, QK2_0=64): d + 16B 2bit 码
//    16     |   B    |    256     |     66      | IQ2_XXS: d + 32×u16（iq2xxs_grid + ksigns_iq2xs）
//    17     |   B    |    256     |     74      | IQ2_XS:  d + 32×u16 + 8B scales
//    22     |   B    |    256     |     82      | IQ2_S:   d + 64B qs + 8B qh + 8B scales
//    18     |   B    |    256     |     98      | IQ3_XXS: d + 96B（iq3xxs_grid + ksigns）
//    21     |   B    |    256     |    110      | IQ3_S:   d + 64B qs + 8B qh + 8B signs + 4B scales
// 透传（family 0）：PLE per_layer_token_embd（type20 按名字排除）、BF16(30)、
//                  Q6_K(14)/Q5_K(13)/IQ4_XS(23)/Q4_K(12) 稠密、F32/F16

// ---- 环境开关（引擎侧）----
// MONSTER_MONPACK=0/1                加载器认边车（默认 1，边车存在才生效）
// MONSTER_MONPACK_TYPES="20,21,..."  灰度白名单（空 = 全部打包类型）
// GGML_CUDA_MONPACK_TC_MIN_M=16      TC 路径 M_total 阈值
// GGML_CUDA_MONPACK_DISABLE=1        kill-switch（强制 GEMV-on-pack）
