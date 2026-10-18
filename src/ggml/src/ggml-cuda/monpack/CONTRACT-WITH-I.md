# 集成代理 I 已完工——monpack.h 必须满足的契约（详见 monpack/monpack.h 汇报要求）
1. monpack.h 必须普通 g++ 可编译（llama 目标的 .cpp 也 include 它）。
2. I 已插好的调用点（ggml-cuda.cu :2701 try_mul_mat(dst,src0) / :2828 try_mul_mat_id(dst) / llama-monpack-loader.cpp:1692 register_tensor）：
   - bool is_registered(const void* dev_ptr)
   - bool try_mul_mat(ggml_tensor* dst, const ggml_tensor* src0)
   - bool try_mul_mat_id(ggml_tensor* dst)   // 需要更多参数就在头文件注释注明并汇报
   - void register_tensor(const ggml_tensor* t)  // t->data=设备端打包布局起点；k_tile 等从 type+ne[] 推导
3. t->data 就是设备端打包布局（I 的实现：blob 字节经 ggml_backend_tensor_set 上传，无双份）。
4. 格式宪法已修订：fam_desc 88B / index 104B / identity u64LE / blob_off 256B 对齐。
5. I 在 needs_sync 对注册张量返回 false——内核必须全程零 host-sync。
