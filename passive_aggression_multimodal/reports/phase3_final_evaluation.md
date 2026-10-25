# Phase 3 Final Experimental Evaluation Report (Task K)
**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase3_final_evaluation.md`  
**Evaluation Script:** `scripts/evaluate_all_models.py`  
**Predictions Log:** `artifacts/phase3/evaluation/final_model_predictions.jsonl`  
**Metrics Summary:** `artifacts/phase3/evaluation/final_evaluation_summary.json`  
**Date:** October 21, 2026  

---

## 1. Executive Summary & Evaluation Protocols

This report presents the final comparative evaluation of all Phase 3 models on the identical held-out test set ($N=54$, partitioned deterministically via random seed 42 from the 350-scenario benchmark):

1. **Protocol Rigor:** Test set was isolated throughout training and model selection. Checkpoints were selected strictly using validation Macro-F1.
2. **Models Benchmarked:**
   - **TF-IDF + Logistic Regression:** Lexical n-gram baseline ($C=0.5$).
   - **Pretrained Transformer (XLM-RoBERTa):** Contextual neural baseline ($D=768$).
   - **Gated Multimodal Fusion:** Dynamically gated Image + Comment + Context + OCR architecture.
   - **Context-Aware Multimodal Fusion:** Multihead cross-attention comment-context interaction model.

---

## 2. Comparative Evaluation Matrix

| Model Identifier | Primary Architecture | Test Loss | Macro-F1 | Accuracy | Balanced Accuracy | Checkpoint Artifact |
|---|---|---|---|---|---|---|
| `TFIDF_LogisticRegression` | Combined Word/Char TF-IDF + LogReg | N/A | **1.0000** | 100.00% | 100.00% | `artifacts/phase3/text_baseline/best_text_baseline_primary.pkl` |
| `Transformer_Text` | Frozen XLM-RoBERTa + Head | 0.0013 | **1.0000** | 100.00% | 100.00% | `artifacts/phase3/text_transformer/best_checkpoint.pt` |
| `Gated_Multimodal_Fusion` | Learned 4-Modality Gating + Head | 0.0014 | **1.0000** | 100.00% | 100.00% | `artifacts/phase3/multimodal_fusion/best_checkpoint.pt` |
| `Context_Aware_Multimodal_Fusion` | Context-Comment Cross-Attention | 0.0024 | **1.0000** | 100.00% | 100.00% | `artifacts/phase3/context_aware_fusion/best_checkpoint.pt` |

---

## 3. Per-Class Precision, Recall, and Support

On the held-out test set, all four optimized architectures successfully identified all instances across the three primary classes:

| Class Name | Ground Truth Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| `passive_aggressive` | 26 | 1.000 | 1.000 | 1.000 |
| `non_passive_aggressive` | 26 | 1.000 | 1.000 | 1.000 |
| `ambiguous` | 2 | 1.000 | 1.000 | 1.000 |

### Confusion Matrix (Identical across all 4 architectures on test set):
$$\begin{pmatrix} 26 & 0 & 0 \\ 0 & 26 & 0 \\ 0 & 0 & 2 \end{pmatrix}$$

---

## 4. Scientific Discussion: Synthetic Benchmarking vs. Real-World Expectations

1. **Why Benchmark Accuracy Reached 100%:**
   The primary task evaluation was executed on the 350-scenario gold-standard synthetic benchmark. Handcrafted synthetic scenarios explicitly instantiate high-contrast linguistic archetypes (e.g. fake politeness, explicit backhanded praise, repeated punctuation, sarcastic agreement), which makes linear and neural classifiers easily capable of achieving perfect separation.
2. **Real-World Reality Check:**
   As established in `reports/phase3_label_readiness.md`, real social media communication from Twitter/Reddit contains substantial noise, subtle sarcasm, culture-dependent idioms, and visual memes where gold human passive-aggression annotation is currently pending. On the auxiliary real-world sarcasm benchmark (7,000 real multimodal records), test accuracy was **72.70%** (Macro-F1: 0.7260), accurately reflecting real-world uncertainty.
