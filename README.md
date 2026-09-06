# LoRA Learn

A minimal, hand-written LoRA demo — build an understanding of LoRA's inner workings from scratch.

No `peft` or other libraries. Just a hand-written LoRA layer plus a complete training loop, using MNIST to show it visually:

1. First pre-train a base network on all 10 MNIST classes → simulating "an already-good base model"
2. Then teach this model "new knowledge / adapt it to a specific subset," comparing two approaches:
   - **A) Full fine-tuning** (every parameter is updated)
   - **B) LoRA fine-tuning** (main network frozen, only the low-rank bypass `A·B` is trained)
3. Key metric: number of trainable parameters (LoRA uses only a few percent) + how the results compare
4. At inference, merge the bypass back into the main weights `W' = W + scale·A·B` and verify zero extra overhead

## Why v2 converges while v1 didn't

- The standard LoRA trick: `B` starts at 0, so the bypass contributes 0 at initialization and the model departs from its "original weights" — this is the whole selling point.
- Why v1 failed: its base had only seen classes 0~4 and was completely "blind" to 5~9. Forcing LoRA to learn from that blind spot (with randomly initialized bypasses) blew up the logits — loss stuck around ~7.9 and was numerically unstable.
- v2 fix: the base has seen all 10 classes and already works; LoRA only makes small corrections, so values stay stable and it reliably converges.

> This mirrors real image/genAI practice: get a good model first, then add LoRA to learn a style.

## Requirements

- Python 3.x
- PyTorch
- torchvision

## Run

```bash
pip install torch torchvision
python demo_lora.py
```

MNIST data is downloaded automatically to `./data` on first run (already in `.gitignore`).

## Core: the hand-written LoRA layer

```python
class LoRALayer(nn.Module):
    """Low-rank bypass that can plug in beside any Linear. Freeze original W, train only A, B."""
    def __init__(self, in_features, out_features, rank=4, alpha=4.0):
        super().__init__()
        # A: small Gaussian init; B: all zeros (start point = original model — the key to LoRA!)
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank          # scaling, controls bypass strength

    def forward(self, x):
        # x: [..., in];  returns [..., out] bypass delta
        return (x @ self.A @ self.B) * self.scale
```

## Takeaways

With only a few percent of the parameters of full fine-tuning, LoRA reaches near-full-fine-tuning quality — and at inference the bypass can be merged back into the main model with no added runtime cost. That's exactly why LoRA dominates in image / large-model workflows: it saves GPU memory, saves storage, and still delivers.

## License

MIT
