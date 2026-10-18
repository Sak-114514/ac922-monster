// monpack.h — monpack v2 CUDA 内核对外契约（给集成代理）
//
// 本头文件必须能被普通 g++ 编译（llama 侧 .cpp 直接 include），因此只用前置声明。
//
// ---- 四个入口与调用时机 ----
//
// 1. void monpack::register_tensor(const ggml_tensor * t);
//       调用时机：加载器（llama-monpack-loader.cpp llama_monpack::register_tensor）在打包 blob
//       字节经 ggml_backend_tensor_set 到达张量最终设备 buffer 之后调用；t->data 即设备端打包
//       布局起点（无第二份拷贝）。family 从 t->type 推导；k_tile 按 monpack_format.h 规则从
//       ne[0] 推导（%256→256、%128→128、%64→64，其余拒绝注册）；ne_k/ne_n/ne_e 取 ne[0..2]。
//       注册期会按需一次性分配 grouped-TC 的 work-list 设备缓冲（任何 graph 捕获之前）。
//    void monpack::register_tensor(const void * tensor_data, uint32_t ggml_type, uint32_t family,
//                                  uint32_t ne_k, uint32_t ne_n, uint32_t ne_e, uint32_t k_tile,
//                                  const void * blob_base);
//       显式形式（同语义；blob_base 仅做指针关系健全性检查，可为 nullptr）。
//
// 2. bool monpack::is_registered(const void * tensor_data);
//       调用时机：任意。指针命中注册表返回 true。
//
// 3. bool monpack::try_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
//                              const ggml_tensor * src0);
//       【必须使用此形式】调用时机：ggml_cuda_mul_mat 内、stock 分派（MMVF/MMF/MMVQ/MMQ/cuBLAS）
//       之前，对 GGML_OP_MUL_MAT 节点探测（当前插入点 ggml-cuda.cu:2706，ctx 就在作用域内）。
//       需要.ctx 的原因：CUDA graph 捕获要求内核与宿主路径用同一个 ctx.stream()；无 ctx 无法
//       恢复正确的流（公共同名重载是显式报错的存根，见下）。
//       命中条件（全部满足才返回 true 并已在 ctx.stream() 上完成计算）：
//        - src0->data 已注册且类型属于 8 个打包类型，src0 为 [ne_k, ne_n, ne_e] 连续布局；
//        - src1 (= dst->src[1]) / dst 为 F32；
//        - M = ne11*ne12*ne13：
//             M <= 8                 -> monpack_gemv（与 stock MMVQ 逐位一致；多列 small_k 形状、
//                                       多 channel/sample 维均支持）
//             8 < M < TC_MIN_M(16)   -> 按 8 列分块的 gemv（仅 ne12==ne13==1 时提供）
//             M >= TC_MIN_M          -> monpack_tc_grouped（fp16 TC + fp32 累加；仅 ne12==ne13==1
//                                       且 M<=65536 时提供）
//        - 环境开关：GGML_CUDA_MONPACK_DISABLE=1 恒 false；
//                    GGML_CUDA_MONPACK_TC_MIN_M=N 设 TC 阈值（默认 16）。
//       未命中返回 false，调用方走原路径（已注册张量的数据是打包布局，stock 内核读不懂——
//       集成侧必须保证注册张量只会走到已覆盖的调用点；fused gate/up+GLU 调用点不在本契约内）。
//
// 4. bool monpack::try_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
//       【必须使用此形式】调用时机：ggml_cuda_mul_mat_id 分派区（当前插入点 ggml-cuda.cu:2832，
//       ctx 就在作用域内）。src0 = dst->src[0]、src1 = dst->src[1]（F32，[K, ne11, ne12]）、
//       ids = dst->src[2]（I32，[n_used, ne12]），语义同 stock mul_mat_id。
//       命中条件：
//        - 注册/类型/形状检查同上（另需 ids->nb[0]==4、ids->ne[1]==dst->ne[2]、ne13==1、ne_e<=512）；
//        - tokens = ne2 = ids->ne[1]：
//             tokens <= 8           -> GEMV MoE 变体（与 stock 逐位一致，含 Volta IQ2_S 专用内核；
//                                      ne2==1 且 GGML_SM70_DECODE_LINEAR_CHUNKS=8 时走 moe 内核，
//                                      复刻 stock 同等条件）
//             8 < M_total < TC_MIN_M -> 按 token 8 分块的 gemv
//             M_total >= TC_MIN_M   -> monpack_tc_grouped：设备侧 work list（graph 内前置
//                                      mp_tc_build_wl kernel 遍历 ids 生成 (e,n_tile,m_e,rows_off) 项，
//                                      缓冲注册期预分配），持久化 80 CTA × 256 线程，零 host-sync、
//                                      零运行期分配，CUDA graph 兼容（ggml-cuda.cu:3793 的
//                                      needs_sync 检查对注册张量恒为 false 路径）；M_total<=65536。
//
// ---- 无 ctx 同名存根 ----
//   bool try_mul_mat(ggml_tensor * dst, const ggml_tensor * src0);
//   bool try_mul_mat_id(ggml_tensor * dst);
//   为兼容已插入的调用点形状而存在；一旦被真正调用（注册张量到达计算）立即 GGML_ABORT 报错，
//   提示调用点改传 ctx——绝不静默落到会误读打包字节的 stock 内核。
//
// ---- 数值契约 ----
//   GEMV 路径（M<=8 及其分块）与 stock MMVQ 内核逐位一致：相同 warp/lane<->k 映射、相同 fp32
//   求和顺序、逐字相同的 vec_dot 代数（仅权重字节寻址换到打包布局）；q8_1 量化输入复用
//   quantize_row_q8_1_cuda 与 ctx.q8_1_cache。TC 路径是 fp16 mma + fp32 累加，代数同构于 CPU
//   反量化真值（ggml-quants.c），不承诺与 stock 逐位一致。

#pragma once

#include <stdint.h>

struct ggml_tensor;
struct ggml_backend_cuda_context; // 定义在 ggml-cuda/common.cuh；此处只需前置声明（g++ 可编译）

namespace monpack {

// registration
void register_tensor(const ggml_tensor * t);
void register_tensor(const void * tensor_data, uint32_t ggml_type, uint32_t family,
                     uint32_t ne_k, uint32_t ne_n, uint32_t ne_e, uint32_t k_tile,
                     const void * blob_base);

bool is_registered(const void * tensor_data);

// dispatch hooks — the ctx overloads are the real entry points
bool try_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * src0);
bool try_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// no-ctx stubs matching the originally inserted call-site shapes; abort loudly if reached
bool try_mul_mat(ggml_tensor * dst, const ggml_tensor * src0);
bool try_mul_mat_id(ggml_tensor * dst);

} // namespace monpack
