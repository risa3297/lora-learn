# LoRA Learn

手写 LoRA 最小 demo —— 从零理解 LoRA 底层机制。

不依赖 `peft` 等库，纯手写 LoRA 层 + 完整训练循环，用 MNIST 直观演示：

1. 先在 MNIST 全部 10 类上预训练 base 网络 → 模拟"一个已经很好用的基座模型"
2. 再给这个模型"教新知识 / 适配特定子集"，对比两种方式：
   - **A) 全量微调**（所有参数都动）
   - **B) LoRA 微调**（冻结主网络，只训低秩旁路 A·B）
3. 关键指标：可训练参数量（LoRA 只有几个百分点）+ 效果对比
4. 推理时把旁路合并回主权重 `W' = W + scale·A·B`，验证零额外开销

## 为什么 v2 会收敛而 v1 不会

- LoRA 标准做法 B 初始为 0，旁路初始贡献为 0，模型从"原权重"出发——这正是 LoRA 的卖点。
- v1 的失败：base 只见过 0~4，对 5~9 完全"盲"，硬让 LoRA 从盲区起步学，旁路 logits 推爆。
- v2 修正：base 见全 10 类已能用，LoRA 只做少量修正，数值稳定必然收敛。

> 这跟真实练图完全一致：先有好模型，再加 LoRA 学风格。

## 环境要求

- Python 3.x
- PyTorch
- torchvision

## 运行

```bash
pip install torch torchvision
python demo_lora.py
```

MNIST 数据会在首次运行时自动下载到 `./data`（已加入 `.gitignore`）。

## 核心：手写 LoRA 层

```python
class LoRALayer(nn.Module):
    """可插在任意 Linear 旁的低秩旁路。冻结原 W，只训练 A、B。"""
    def __init__(self, in_features, out_features, rank=4, alpha=4.0):
        super().__init__()
        # A 高斯小初始化; B 全 0 (保证训练起点 = 原模型, LoRA 的关键!)
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank          # 缩放, 控制旁路强度

    def forward(self, x):
        # x: [..., in];  返回 [..., out] 的旁路增量
        return (x @ self.A @ self.B) * self.scale
```

## 结论

LoRA 用全量微调几个百分点的参数，就能达到接近全量微调的效果，且推理时权重可合并回主模型、不增加任何运行时开销——这正是它在图像 / 大模型领域流行的根本原因：省显存、省存储、效果还不差。

## License

MIT
