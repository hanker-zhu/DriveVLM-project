## Speculative Decoding

调查了Self-Speculative Decoding后, 发现其具体实现依赖于“经验性”的手动指定对哪些层进行筛选, 难以复现。

所以采用原始的Speculative Decoding进行计算, 使用量化后的模型作为draft model。

实际效果在和原始模型保持一致的情况下达到了接近量化后模型的效率

具体实现见src/reference.py