# Classification Head Design & Mathematical Formulation Report
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Module:** `src/models/classification_head.py`  
**Test Suite:** `tests/test_classification_head.py`  
**Date:** October 21, 2026  

---

## 1. Architectural Overview

The `ClassificationHead` is a modular PyTorch neural module that maps fused multimodal (or unimodal) feature representations $\mathbf{h} \in \mathbb{R}^{D_{\text{in}}}$ to class logits $\mathbf{z} \in \mathbb{R}^{C}$, where $C=3$ for the primary passive-aggression classification task.

### Forward Pipeline:
$$\mathbf{h}^{(1)} = \text{LayerNorm}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1)$$
$$\mathbf{h}^{(2)} = \text{Dropout}(\text{GELU}(\mathbf{h}^{(1)}), p=0.1)$$
$$\mathbf{z} = \mathbf{W}_2 \mathbf{h}^{(2)} + \mathbf{b}_2$$

Where:
- $\mathbf{h} \in \mathbb{R}^{D_{\text{in}}}$ is the input embedding ($D_{\text{in}} = 512$ for gated fusion or $768$ for transformer baselines).
- $\mathbf{W}_1 \in \mathbb{R}^{D_{\text{hidden}} \times D_{\text{in}}}$ and $\mathbf{b}_1 \in \mathbb{R}^{D_{\text{hidden}}}$ with $D_{\text{hidden}} = 256$.
- $\text{LayerNorm}$ provides feature stabilization across heterogeneous multimodal scales.
- $\text{GELU}$ activation ensures non-linear capacity with smooth gradients.
- $\text{Dropout}(p=0.1)$ provides regularization against overfitting.
- $\mathbf{W}_2 \in \mathbb{R}^{C \times D_{\text{hidden}}}$ and $\mathbf{b}_2 \in \mathbb{R}^{C}$ project to class logits ($C=3$).

---

## 2. Explicit Class Mapping & Logit Semantic Grounding

The class mapping is hardcoded and saved in model checkpoints to prevent semantic drift:

```python
PRIMARY_LABEL_TO_IDX = {
    "passive_aggressive": 0,
    "non_passive_aggressive": 1,
    "ambiguous": 2,
}

PRIMARY_IDX_TO_LABEL = {
    0: "passive_aggressive",
    1: "non_passive_aggressive",
    2: "ambiguous",
}
```

For auxiliary sarcasm detection, an identical head structure uses:
```python
SARCASM_LABEL_TO_IDX = {
    "sarcastic": 0,
    "non_sarcastic": 1,
    "ambiguous": 2,
}
```

---

## 3. Loss Formulation & Probability Estimation

### 3.1 Primary Objective Function
The classification head optimizes multi-class Cross-Entropy Loss over unnormalized logits $\mathbf{z}$:

$$\mathcal{L}_{\text{CE}}(\mathbf{z}, y) = - \log \left( \frac{\exp(z_y)}{\sum_{c=0}^{C-1} \exp(z_c)} \right) = - z_y + \log \left(\sum_{c=0}^{C-1} \exp(z_c)\right)$$

With optional class weighting $\mathbf{w} \in \mathbb{R}^C$ to counter label imbalance:
$$\mathcal{L}_{\text{wCE}}(\mathbf{z}, y) = - w_y \log \left( \frac{\exp(z_y)}{\sum_{c=0}^{C-1} \exp(z_c)} \right)$$

### 3.2 Probability Prediction & Abstention Mechanism
Probabilities are extracted strictly via post-logit Softmax:
$$P(Y = c \mid \mathbf{h}) = \frac{\exp(z_c)}{\sum_{j=0}^{C-1} \exp(z_j)}$$

Default prediction selects the mode:
$$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} P(Y = c \mid \mathbf{h})$$

When an uncertainty threshold $\tau \in [0.33, 1.0)$ is configured, an **abstention policy** is activated:
$$\hat{y}_{\tau} = \begin{cases} 
\arg\max_c P(Y=c \mid \mathbf{h}), & \text{if } \max_c P(Y=c \mid \mathbf{h}) \ge \tau \\
2 \text{ (ambiguous)}, & \text{otherwise}
\end{cases}$$

This guarantees that instances where the model exhibits near-uniform uncertainty are routed directly to the `ambiguous` category rather than making forced uncalibrated binary decisions.

---

## 4. Verification and Validation Results

The implementation in `src/models/classification_head.py` was validated against unit tests in `tests/test_classification_head.py`:
1. **Shapes:** Verified output shape $[B, 3]$ for $B \in \{1, 4, 8\}$.
2. **Gradient Flow:** Forward and backward loss computation demonstrated valid non-zero finite gradients for all weights and biases.
3. **Robustness:** Throws explicit `ValueError` when provided with `NaN`/`Inf` inputs, feature dimension mismatches, or out-of-bounds target IDs.
4. **Checkpoint Integrity:** Serializes weights, architecture dimensions, and label mappings to `.pt` format; reloaded weights reproduce exact numerical outputs ($< 10^{-5}$ tolerance in evaluation mode).
