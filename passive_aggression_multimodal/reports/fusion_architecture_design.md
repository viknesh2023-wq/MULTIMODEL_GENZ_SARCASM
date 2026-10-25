# Multimodal Fusion Architecture Design: Context-Aware Passive-Aggression Detection

## 1. Architectural Overview & Research Motivation

Passive-aggression in multimodal social media is an indirect pragmatic phenomenon characterized by a divergence between explicit surface sentiment and underlying pragmatic hostility. Detecting it reliably requires synthesizing clues across four distinct modalities:
1. **Target Comment**: The primary linguistic utterance under evaluation.
2. **Visual Content**: Meme imagery, facial expressions, screenshot context, or contextual scenes.
3. **Conversation Thread / Context**: Preceding parent remarks or dialogue history that establish interpersonal dynamics.
4. **Image OCR Text**: Embedded textual slogans, captions, or dialogue bubbles within the visual asset.

Simply concatenating raw embeddings does not constitute a passive-aggression detector. A trainable classification head with task-specific multimodal gating must learn pragmatic contrast patterns (e.g., superficially polite phrasing paired with visual ridicule or condescending context).

```
[Target Comment]  --> [DeBERTa/RoBERTa (768)] --> [Projection (512)] ---\
[Visual Scene]    --> [CLIP/SigLIP     (768)] --> [Projection (512)] ---+--> [Modality Attention] --> [Fused (512)]
[Thread Context]  --> [DeBERTa/RoBERTa (768)] --> [Projection (512)] ---|           Gate
[Embedded OCR]    --> [DeBERTa/RoBERTa (768)] --> [Projection (512)] ---/
                                                                                  |
                                                                        +---------+---------+
                                                                        |                   |
                                                                  [Primary Head]     [Auxiliary Head]
                                                                Passive-Aggression       Sarcasm
                                                                   (3 Classes)         (3 Classes)
```

---

## 2. Mathematical Formulation & Modality Projections

Let the input record be denoted by $X = (x_{comment}, x_{image}, x_{context}, x_{ocr})$ with presence mask indicators $M = (m_{image}, m_{context}, m_{ocr}) \in \{0, 1\}^3$. $x_{comment}$ is guaranteed to be present ($m_{comment} = 1$).

### 2.1 Feature Representation
Each modality is encoded into a high-dimensional dense vector:
$$e_c = \text{Encoder}_{text}(x_{comment}) \in \mathbb{R}^{d_t}$$
$$e_v = \text{Encoder}_{img}(x_{image}) \in \mathbb{R}^{d_v} \quad \text{if } m_v = 1 \text{ else } \mathbf{0}$$
$$e_x = \text{Encoder}_{text}(x_{context}) \in \mathbb{R}^{d_t} \quad \text{if } m_x = 1 \text{ else } \mathbf{0}$$
$$e_o = \text{Encoder}_{text}(x_{ocr}) \in \mathbb{R}^{d_t} \quad \text{if } m_o = 1 \text{ else } \mathbf{0}$$
where $d_t = 768$ (DeBERTa-v3-base) and $d_v = 768$ (CLIP-ViT-B/32 or SigLIP-B/16).

### 2.2 Modality-Specific Projection
Each raw representation is mapped into a shared latent space $\mathbb{R}^H$ (where $H = 512$):
$$h_c = \text{GELU}(\text{LayerNorm}(W_c e_c + b_c))$$
$$h_v = m_v \cdot \text{GELU}(\text{LayerNorm}(W_v e_v + b_v))$$
$$h_x = m_x \cdot \text{GELU}(\text{LayerNorm}(W_x e_x + b_x))$$
$$h_o = m_o \cdot \text{GELU}(\text{LayerNorm}(W_o e_o + b_o))$$

### 2.3 Dynamic Modality-Aware Gating
To prevent absent or uninformative modalities from injecting noise, a gating network calculates modality attention weights:
$$z = [h_c \,\|\, h_v \,\|\, h_x \,\|\, h_o] \in \mathbb{R}^{4H}$$
$$\alpha_{raw} = \text{Softmax}(W_g z + b_g) \in \mathbb{R}^4$$
$$\alpha_k = \frac{\alpha_{raw, k} \cdot m_k}{\sum_{j} \alpha_{raw, j} \cdot m_j + \epsilon}, \quad k \in \{c, v, x, o\}$$
The fused multimodal vector is then produced:
$$h_{fused} = \text{GELU}\left(\text{LayerNorm}\left(W_f [\alpha_c h_c \,\|\, \alpha_v h_v \,\|\, \alpha_x h_x \,\|\, \alpha_o h_o] + b_f\right)\right) \in \mathbb{R}^H$$

---

## 3. Classification Heads

The model supports multi-task prediction:
1. **Primary Passive-Aggression Head**:
   $$\hat{y}_{pa} = \text{Softmax}(W_{pa2} \cdot \text{ReLU}(W_{pa1} h_{fused} + b_{pa1}) + b_{pa2}) \in \mathbb{R}^3$$
   Classes:
   - `passive_aggressive` (index 0)
   - `non_passive_aggressive` (index 1)
   - `ambiguous` (index 2)

2. **Auxiliary Sarcasm Head**:
   $$\hat{y}_{sarcasm} = \text{Softmax}(W_{s2} \cdot \text{ReLU}(W_{s1} h_{fused} + b_{s1}) + b_{s2}) \in \mathbb{R}^3$$
   Classes:
   - `sarcastic` (index 0)
   - `non_sarcastic` (index 1)
   - `ambiguous` (index 2)

3. **Pragmatic Subtype Head** (Prepared for fine-grained analysis):
   $$\hat{y}_{subtype} \in \mathbb{R}^7$$
   Classes:
   - `fake_politeness`
   - `indirect_criticism`
   - `backhanded_compliment`
   - `implied_blame`
   - `sarcastic_agreement`
   - `dismissiveness`
   - `other_or_unclear`

---

## 4. Phase 2 Verification & Shape Testing

The architecture is implemented in [`src/preprocessing/fusion_interface.py`](file:///d:/multimodel_passive_aggression/src/preprocessing/fusion_interface.py) and validated with batch sizes $B \in \{1, 4, 8\}$ across missing-modality configurations:

| Modality Config Tested | Comment | Image | Context | OCR | Logits Output Shape | Status |
|---|---|---|---|---|---|---|
| Complete Multimodal | Present | Present | Present | Present | `[B, 3]`, `[B, 3]`, `[B, 7]` | Passed |
| Text + Image Only | Present | Present | Absent | Absent | `[B, 3]`, `[B, 3]`, `[B, 7]` | Passed |
| Text + Context Only | Present | Absent | Present | Absent | `[B, 3]`, `[B, 3]`, `[B, 7]` | Passed |
| Unimodal Text Only | Present | Absent | Absent | Absent | `[B, 3]`, `[B, 3]`, `[B, 7]` | Passed |

**Phase 2 Boundary Adherence**:
No model training, backpropagation, or parameter updates have been performed in Phase 2. The fusion architecture remains an un-fine-tuned, shape-validated modular interface ready for Phase 3 training.
