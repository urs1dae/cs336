# TO DO

- [ ] 优化 BPE 训练阶段 merge 统计更新：当前一次 merge 会对受影响序列执行“整序列删统计 + 整序列加统计”（`assignment1-basics/cs336_basics/tokenizer.py` + `assignment1-basics/cs336_basics/bpe_ops.py`），存在大量重复计算；改为仅更新 merge 发生位置附近窗口的增量统计。
- [ ] 优化 `pair_max_heap` 的 stale 节点问题：当前 lazy 删除 + 每次计数变化都 `heappush`（`assignment1-basics/cs336_basics/bpe_ops.py`）会累积 stale entry；引入 version 戳校验，或按 stale 比例阈值触发 heap 重建。
- [ ] 重构 `pair_to_sequences` 数据结构：当前存 `set[ByteSeq]`（`assignment1-basics/cs336_basics/bpe_ops.py`）哈希开销较高；引入 `seq_id: int`，改为 `pair -> set[int]`，并用数组/紧凑结构维护 `id -> sequence/count` 映射。
- [ ] 优化预分词热点：`re.findall` 高频调用（`assignment1-basics/cs336_basics/pretokenize.py`）改为全局预编译 `re.compile(...)`；将 `special_token_set = set(...)` 外提或缓存，避免重复构建。
