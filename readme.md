# Latent Prototype Routing for Mixture-of-Experts (LPR)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

Official implementation of **[Latent Prototype Routing: Achieving Near-Perfect Load Balancing in Mixture-of-Experts](https://arxiv.org/)**  

*Work in progress*

> 📧 If you want to discuss with me or knowing the progress,**Contact**: jiajie.y@wustl.edu  
> ✨ **Key Contribution**: Near-perfect expert load balancing with minimal performance tradeoffs

## Experimental Results (1B Tokens on FineWeb-100BT)
| Model                      | Test Loss ↓ | GINI ↓   | Min-Max ↑ |
|----------------------------|-------------|----------|-----------|
| **Mixtral-0.6B (128-8)**   | **3.683**   | 0.635    | 3.33e-6   |
| Mixtral-LPR-0.6B (w/o init)| 3.747       | **0.047**| **0.649** |
| **DeepSeekV3-0.6B (128-8)**| **3.673**   | 0.790    | 6.41e-9   |
| DeepSeekMoe-LPR (w/o init) | 3.720       | **0.036**| **0.724** |
| **Qwen3Moe-0.6B (128-8)**  | **3.666**   | 0.707    | 1.27e-16  |
| Qwen3Moe-LPR (w/ init)     | 3.685       | 0.057    | 0.597     |
| Qwen3Moe-LPR (w/o init)    | 3.697       | **0.039**| **0.696** |

*Metrics explanation*:  
- ↓ Lower is better | ↑ Higher is better
- **GINI**: Load imbalance (0 = perfect balance)
- **Min-Max**: Ratio of least/most used experts

---

## Installation

### 1. Replicating Paper Results
We provide modified implementations for:
- Mixtral
- Qwen3Moe
- DeepseekV3

**Steps**:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Apply custom transformer components
cp -r ModifiedTransformsLibFile/* $(python -c "import site; print(site.getsitepackages()[0])")/transformers/

# 3. Prepare datasets
mkdir -p data/{c4val,fineweb}
# Download C4 validation to data/c4val
# Download FineWeb-100BT to data/fineweb/100BT

# 4. Run training & evaluation
./train.sh
./eval_all.sh
```

## 2. Using LPR Router in Your Project
Minimal integration steps:

1. Copy router files:

```bash
cp model_zoo/router.py your_project/
cp model_zoo/Metrics.py your_project/
```
2. Replace existing router in PyTorch model:
```python
from router import TokenDistributionRouter

class YourMoEBlock(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        # Replace existing router with:
        self.router = TokenDistributionRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            k=top_k
        )
    
    def forward(self, hidden_states):
        # Routing logic
        routing_weights, kl, selected_experts, router_logits, weighted_key = self.router(hidden_states)
        # ... rest of forward pass

```

## License
Apache License 2.0 - See LICENSE for details.

Attribution Requirement: Derivative works must retain proper attribution to the original authors.

💡 Contact for Collaboration: For implementation questions or research collaboration opportunities, email jiajie.y@wustl.edu
