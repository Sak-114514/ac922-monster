# 引擎源码 git 历史说明

- 本仓库 `src/` 是服务器 monster 树（HEAD=1e542c5 "sweep: cut foreign cpu archs..."）
  加上截至 2026-10-18 的全部未提交差量的**完整文件快照**。
- 差量本身见 `src-uncommitted-deltas-vs-1e542c5-20261018.patch`（7422 行，64 个文件）。
- 逐提交谱系（cut1-cut4 裁剪、monster-v0 建分支等）保留在服务器工作树
  `~/ac922-monster/src/.git`（monster-v0 分支），完整 bundle 需在服务器先补全
  blob（其 origin 为 blob:none 部分克隆）后导出。
- 谱系要点：1afa5654（上游 winner 基线，单流 1350.8 PP / 150.7 TG）→ cut1-cut4 裁剪
  （852k→427k 行，每步过构建+逐位回归）→ monster-v0 → 战役期 300+ 未提交演进
  （图组合/四 bank/分片等，演进过程由 docs/codex-notes 与 docs/fixtures-records 全程记录）。
