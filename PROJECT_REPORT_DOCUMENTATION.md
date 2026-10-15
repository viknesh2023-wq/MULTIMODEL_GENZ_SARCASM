# A Multimodal Vision-Language Framework for Gen-Z Sarcasm Detection using XLM-R, CLIP and Vision-Language Models

**Project Topic:** A Multimodal Vision-Language Framework for Gen-Z Sarcasm Detection using XLM-R, CLIP and Vision-Language Models  
**Document Type:** Formal Project Report & Research Documentation  
**Typography Guidelines:** Headings: Bold, Font Size 12, Times New Roman | Sub-headings: Font Size 12, Times New Roman | Paragraphs: Font Size 12, Times New Roman  
**Execution Environment:** Windows 11, Python 3.12.10 (`internvl_env`), PyTorch 2.11.0+cu128, CUDA 12.8  
**Compute Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (5.997 GB Dedicated VRAM, 24.2 GB System RAM)  
**Project Status:** 100% Implemented, Trained, Evaluated, and Production Operational  

---

## CONTENTS

1. **Project Topic**
2. **Abstract** *(Strictly within 300 words)*
3. **Introduction** *(Comprehensive Foundation — Minimum of 1 Page)*
   - 1.1 Evolution of Digital Communication and Sarcasm
   - 1.2 Figurative Complexity of Sarcasm in Social Computing
   - 1.3 The Gen-Z Linguistic Shift: Slang, Acronyms, and Emoji Dissonance
   - 1.4 Code-Mixing in Multilingual Online Communities
   - 1.5 Cross-Modal Incongruity as the Core Mechanism of Meme Sarcasm
   - 1.6 Limitations of Prior Unimodal and Bimodal Approaches
   - 1.7 Research Objectives and Scope
   - 1.8 Primary Scientific Contributions
4. **Background Study** *(Related Papers and Literature Survey)*
   - 2.1 Classical and Lexical Approaches to Text Sarcasm Detection
   - 2.2 Deep Contextual and Pretrained Transformer Architectures
   - 2.3 Multimodal Sarcasm, Irony, and Humor Detection
   - 2.4 Contrastive Vision-Language Learning and Foundation Models
   - 2.5 Code-Mixed and Dravidian-Language Social Computing
   - 2.6 Fine-Grained Affective and Emotion Modeling
   - 2.7 Explainable Artificial Intelligence (XAI) in Natural Language Processing
   - 2.8 Critical Gaps Identified in Literature
5. **Methodology** *(Algorithms, Methods, Datasets — Minimum of 2 Pages)*
   - 3.1 Comprehensive Dataset Architecture and Engineering (1.22M+ Rows)
     - 3.1.1 Text Sarcasm Master Dataset (1,005,167 Rows)
     - 3.1.2 Multimodal Sarcasm Master Dataset (33,077 Rows and Images)
     - 3.1.3 Sentiment Master Dataset (132,324 Rows)
     - 3.1.4 Emotion Master Dataset (54,263 Rows, GoEmotions)
     - 3.1.5 Zero-Leakage Split Isolation and Partitioning Protocols
   - 3.2 Linguistic Preprocessing and Slang Normalization Pipeline
   - 3.3 Feature Extraction Methodologies and Modality Towers
     - 3.3.1 Branch 1: Contextual Text Encoder (XLM-R, 768-D)
     - 3.3.2 Branch 2: Visual Representation Model (CLIP ViT-B/32, 512-D)
     - 3.3.3 Branch 3: Vision-Language Model (InternVL3-2B, 1,536-D)
     - 3.3.4 Branch 4: Sentiment Valence Distribution (XLM-R, 3-D)
     - 3.3.5 Branch 5: Fine-Grained Emotion Spectrum (XLM-R, 28-D)
   - 3.4 Mathematical Formulations and Optimization Objectives
   - 3.5 Assembly of the Unified 2,847-Dimensional Feature Vector
6. **Proposed Model** *(Diagrams and Deep Explanation — Minimum of 2 Pages)*
   - 4.1 End-to-End System Architecture Overview
   - 4.2 Architectural Diagrams (Structural Flow & Pipeline Schematic)
   - 4.3 Deep Multimodal Fusion Classifier (MultimodalFusionMLP)
   - 4.4 Local Explainable AI (XAI) Cognitive Reasoner
   - 4.5 Concurrency Control, Mutual Exclusion, and VRAM Optimization
7. **Results and Discussion** *(Empirical Findings — Minimum of 2 Pages)*
   - 5.1 Experimental Setup and Environmental Constraints
   - 5.2 Performance of Individual Component Models
     - 5.2.1 XLM-RoBERTa Text Sarcasm Classifier Results
     - 5.2.2 XLM-RoBERTa 3-Class Sentiment Classifier Results
     - 5.2.3 XLM-RoBERTa 28-Class GoEmotions Classifier Results
   - 5.3 Performance of the Proposed Multimodal Fusion Classifier
   - 5.4 The 8-Way Modality Ablation Benchmark
   - 5.5 Computational Latency, Throughput, and Hardware Profiling
   - 5.6 Sample Source Code Implementation
     - 5.6.1 MultimodalFusionMLP PyTorch Neural Network Definition
     - 5.6.2 End-to-End Multimodal Extraction and Slicing Routine
     - 5.6.3 CPU-Pinned Local XAI Reasoner with Fallback Engine
     - 5.6.4 Production FastAPI Multimodal Route
   - 5.7 Screenshots and User Interface Module Walkthrough (All 10 Modules)
   - 5.8 Qualitative Case Studies and Semantic Incongruity Analysis
   - 5.9 Error Analysis and Failure Cases
8. **Conclusion**
9. **References** *(Harvard Style, Alphabetically Sorted by Author)*

---

## ABSTRACT

*(Word count: 278 words — Strictly adhering to the maximum limit of 300 words)*

Sarcasm is an inherently complex pragmatic phenomenon where the intended communicative meaning directly contradicts the literal interpretation. In modern social media, particularly among Generation Z, sarcasm is rarely conveyed through isolated text. Instead, it relies on cross-modal incongruity—sharp divergence between surface text and visual context—amplified by internet slang, acronyms, emoji reversals, and multilingual code-mixing (Tamil-English, Kannada-English, Malayalam-English, and Hinglish). Traditional unimodal natural language processing models struggle in these environments because they cannot perceive visual context or non-standard linguistic markers.

This research presents a multimodal vision-language framework for Gen-Z sarcasm detection. The proposed architecture integrates five distinct representation branches into a canonical 2,847-dimensional feature space: (1) 768-D contextual text embeddings from a fine-tuned XLM-RoBERTa model; (2) 512-D visual scene embeddings from OpenAI CLIP ViT-B/32; (3) 1,536-D deep semantic tokens from the InternVL3-2B vision-language tower; (4) 3-D sentiment valence probabilities; and (5) 28-D continuous affective distributions from an XLM-RoBERTa GoEmotions classifier. A 4-layer non-linear Multimodal Fusion Multi-Layer Perceptron (MLP) with Mish activations and train-isolated feature scaling classifies instances into sarcastic and non-sarcastic categories. Furthermore, a local, CPU-pinned Explainable AI (XAI) reasoner translates high-dimensional predictions into evidence-grounded natural-language explanations without GPU VRAM overhead.

Trained on 25,384 multimodal instances and evaluated on an untouched test split of 3,802 samples, the proposed model achieves **79.62% Test Accuracy**, **78.64% Macro F1**, and **83.21% Sarcastic F1**, operating with a forward latency of 0.104 ms. A rigorous 8-way ablation study demonstrates that combining all five modalities yields a **+16.66% Macro F1 improvement** over unimodal text baselines, proving that cross-modal visual grounding and affective modeling are indispensable for modern sarcasm detection.

---

## 1. INTRODUCTION

### 1.1 Evolution of Digital Communication and Sarcasm
Over the past two decades, computer-mediated communication has evolved from synchronous text-based messaging into complex, multimodal social networking environments. Platforms such as Twitter (X), Reddit, Instagram, TikTok, and Discord serve as global digital public spheres where billions of multimodal tokens—combining text, imagery, animated GIFs, audio clips, and hyperlinks—are generated daily. Within this linguistic ecosystem, sarcasm and irony have become ubiquitous rhetorical devices used for humor, social critique, community bonding, defense mechanisms, and political commentary.

In classical rhetoric and linguistics, sarcasm is defined as a form of indirect speech act wherein the speaker utters a statement $S$ with a surface polarity $P_{\text{literal}}$, while intending the listener to infer a contradictory meaning $M$ with an underlying polarity $P_{\text{intended}}$, such that $\text{sign}(P_{\text{literal}}) \ne \text{sign}(P_{\text{intended}})$. For instance, an individual standing in a torrential downpour who states, *"What lovely weather we are having,"* conveys a negative evaluation despite employing purely positive lexical items (*"lovely"*). For automated artificial intelligence systems, detecting such implicit reversals represents one of the most difficult challenges in natural language understanding, as surface syntactic and semantic parsers frequently misclassify sarcastic mockery as sincere praise.

### 1.2 Figurative Complexity of Sarcasm in Social Computing
From a computational perspective, sarcasm detection differs fundamentally from standard sentiment analysis or topic classification. Standard text classifiers operate under the assumption of compositionality: the sentiment of a sentence can generally be derived from the semantic aggregation of its constituent words. In figurative language, however, compositionality breaks down completely. The true sentiment is not encoded within the words themselves, but within the pragmatic tension between the utterance and the context in which it is situated.

When contextual indicators are missing, even human annotators display relatively low inter-rater agreement ($0.60 \le \kappa \le 0.75$) when classifying standalone sarcastic text. In social media streams, this ambiguity is further exacerbated by character limits, absence of vocal inflection (pitch, prosody, stress), lack of facial micro-expressions, and rapid shifts in socio-cultural reference frames. Consequently, conventional machine learning algorithms that rely strictly on n-grams, bag-of-words, or surface sentiment lexicons suffer from severe false-positive and false-negative rates when deployed on real-world digital streams.

### 1.3 The Gen-Z Linguistic Shift: Slang, Acronyms, and Emoji Dissonance
The challenge of automated sarcasm detection has intensified with Generation Z (individuals born between 1997 and 2012) emerging as the primary creators of online social content. Gen-Z digital discourse has established a distinct linguistic sub-dialect characterized by rapid lexical innovation, heavy reliance on ironic detachment, intentional anti-humor, absurdism, and semantic inversion. 

In Gen-Z parlance, words frequently adopt meanings opposite to their conventional definitions:
- The verb phrase *"to cook"* or *"let him cook"* signifies performing exceptionally well or articulating a profound argument.
- Conversely, *"bro thought he cooked"* denotes an embarrassing failure or delusional confidence.
- The adjective *"fire"* denotes high quality, while *"dead"* or the skull emoji (`💀`) indicates overwhelming amusement rather than physical mortality.
- Acronyms such as *"FR"* (for real), *"ONG"* (on God), *"NGL"* (not gonna lie), and *"IYKYK"* (if you know, you know) serve as epistemic markers that modulate pragmatic force.

Crucially, emojis no longer function merely as digital embellishments; they serve as structural punctuation and tonal inversion flags. A statement accompanied by a laughing-crying emoji (`😂`) may represent sincere amusement, whereas the exact same statement appended with a skull emoji (`💀`), a neutral face (`😐`), or a crying face (`😭`) frequently signals deadpan mockery, cynical exhaustion, or biting sarcasm. Traditional Natural Language Processing (NLP) models, trained on formal journalistic or encyclopedic corpora (such as Wikipedia or the Wall Street Journal), fail catastrophically when evaluating these dynamic vernacular structures.

### 1.4 Code-Mixing in Multilingual Online Communities
In multilingual societies—most prominently within South Asia—digital communication is rarely conducted in a single, standardized language. Instead, users routinely engage in linguistic **code-mixing** and **code-switching**, weaving together English with indigenous languages using the Latin script. In Indian social networks, millions of posts blend English with Dravidian languages (Tamil, Kannada, Malayalam, Telugu) and Indo-Aryan languages (Hindi).

Examples of code-mixed social interactions include:
- Tamil-English (*Tanglish*): *"Bro inniku class semma boring da 💀"* (translates to: *"Bro, today's class is extremely boring, mate"*).
- Hindi-English (*Hinglish*): *"Bhai ye presentation toh ekdum next level disaster hai 😂"* (translates to: *"Brother, this presentation is an absolute next-level disaster"*).

Code-mixed text introduces severe computational hurdles:
1. **Absence of Standardized Orthography:** Because Romanized vernacular languages lack standardized spelling conventions, a single word may appear in dozens of phonetic permutations (e.g., the Tamil word *semma* may be written as *sema*, *semmah*, or *cemba*).
2. **Polysemy Across Language Boundaries:** Tokens that possess benign meanings in English may function as derogatory or sarcastic terms in another language.
3. **Out-of-Vocabulary (OOV) Rate:** Monolingual language models (such as BERT or DeBERTa) encounter massive OOV token rates when processing Latinized Indic text, resulting in severe token fragmentation and loss of contextual coherence.

Addressing modern sarcasm therefore strictly requires multilingual foundation encoders—specifically **XLM-RoBERTa**—capable of mapping cross-lingual subwords into a shared contextual embedding space.

### 1.5 Cross-Modal Incongruity as the Core Mechanism of Meme Sarcasm
While textual nuance is vital, the definitive communicative medium of modern social platforms is the **internet meme**—a composite artifact pairing text with an image. In multimodal memes, sarcasm is rarely localized in either the text or the image in isolation. Instead, sarcasm emerges dynamically from the **semantic incongruity** between the two modalities.

Consider an authentic benchmark instance: an image depicts a severely charred, blackened pizza reduced to carbon ashes, accompanied by the text: *"Best pizza I have ever made! 🍕✨"*.
- Evaluated as **text alone**, an NLP model observes high-sentiment tokens (*"best"*, *"ever"*) and food emojis (*"🍕"*, *"✨"*), predicting a strongly positive, non-sarcastic culinary review.
- Evaluated as an **image alone**, a computer vision model detects a plate with burnt food remnants, recognizing a cooking failure without understanding humor or intent.
- Only when an AI system **fuses both modalities** can it perceive the stark contradiction: the text claims supreme culinary success, while the image displays unpalatable wreckage. The resolution of this cross-modal contradiction is the direct identification of sarcasm.

Prior research that treats vision and language as separate pipelines or relies on simplistic early/late fusion mechanisms fails to capture these fine-grained cross-modal incongruities, leaving modern multimodal sarcasm detection largely unsolved.

### 1.6 Limitations of Prior Unimodal and Bimodal Approaches
Existing literature in automated sarcasm detection suffers from four major systemic shortcomings:
1. **Unimodal Linguistic Bias:** The vast majority of published models evaluate text exclusively, rendering them blind to the visual scene context that dictates the meaning of internet memes.
2. **Monolingual Western Centricity:** Benchmark datasets (such as SARC or Twitter Sarcasm Corpus) are overwhelmingly composed of standard English text from Western geopolitical spheres, entirely neglecting Dravidian and Indic code-mixed communication.
3. **Coarse Bimodal Fusion:** Prior multimodal models typically concatenate global image features (e.g., ResNet or basic ViT) with textual embeddings, failing to incorporate auxiliary affective dimensions such as sentiment polarity and multi-label psychological emotions.
4. **Opaque Black-Box Predictions:** Deep multimodal classifiers generate scalar probabilities without providing transparent, evidence-based reasoning. In safety-critical moderation or content intelligence settings, a prediction without interpretable justification cannot be audited or trusted.

### 1.7 Research Objectives and Scope
To resolve these limitations, this research formulates, develops, and empirically evaluates a unified, heterogeneous multimodal intelligence framework designed specifically for Gen-Z communication and cross-modal sarcasm.

The explicit research objectives are:
1. **Massive Multilingual Text Pre-training:** Train and calibrate a multilingual Transformer model (XLM-RoBERTa Base) on over 1,000,000 text sarcasm samples spanning English and code-mixed Dravidian dialects.
2. **Heterogeneous Visual Extraction:** Develop dual visual pipelines combining holistic scene embeddings (OpenAI CLIP ViT-B/32) with fine-grained semantic representations from an advanced open-source vision-language model (OpenGVLab InternVL3-2B).
3. **Affective Incongruity Modeling:** Fine-tune dedicated auxiliary neural heads to extract 3-dimensional sentiment distributions and 28-dimensional multi-label emotion spectrums (GoEmotions), feeding soft psychological states directly into the fusion layer.
4. **Unified 2,847-Dimensional Fusion Space:** Design and optimize a deep non-linear Multimodal Fusion MLP that projects all five representation streams into a cohesive decision boundary.
5. **Rigorous Empirical Ablation:** Execute an exhaustive 8-way modality ablation benchmark on identical held-out test records to quantify the exact mathematical contribution of every feature branch.
6. **Local, Zero-Expense Explainability:** Engineer an Explainable AI (XAI) reasoner grounded in visual question answering and local LLM synthesis, strictly isolated to host CPU to eliminate GPU VRAM overhead and cloud API costs.
7. **Production System Integration:** Package the research pipeline into an operational FastAPI backend with mutual exclusion locking and a full-featured React 18 dashboard.

### 1.8 Primary Scientific Contributions
The concrete scientific and engineering contributions of this work are summarized as follows:
- **Comprehensive Dataset Architecture (1.22M+ Rows):** Synthesized and rigorously audited four master datasets with zero duplicate leakage, including 1,005,167 text sarcasm samples, 33,077 verified multimodal image-text pairs, 132,324 sentiment records, and 54,263 multi-label emotion records.
- **Novel 5-Branch Heterogeneous Fusion:** Engineered a 2,847-dimensional representation space unifying contextual text (768-D), visual scene (512-D), visual semantics (1,536-D), sentiment valence (3-D), and fine-grained emotion (28-D).
- **Superior Empirical Performance:** The proposed Multimodal Fusion MLP achieves **79.62% Test Accuracy**, **78.64% Macro F1**, and **83.21% Sarcastic F1** on held-out test data, outperforming the unimodal text baseline by **+16.66% Macro F1**.
- **First Systematic 8-Way Multimodal Ablation:** Proved mathematically that while affective features improve text F1 by +3.84%, visual grounding provides a massive +15.95% surge, confirming cross-modal incongruity as the primary driver of meme sarcasm.
- **Zero-VRAM Explainable AI Engine:** Designed a dual-tier local reasoning module combining InternVL3-2B visual grounding with an Ollama `llama3.2:1b` CPU engine and a deterministic cognitive fallback (<1 ms latency), guaranteeing 100% service uptime and 0.00 MB GPU VRAM delta on consumer hardware.

---

## 2. BACKGROUND STUDY

### 2.1 Classical and Lexical Approaches to Sarcasm Detection
Early computational research in sarcasm detection approached the task through the lens of rule-based linguistics and feature engineering. Pang and Lee (2008) established foundational techniques for sentiment analysis, noting that figurative expressions represent a major stumbling block for polarity classifiers. Davidov, Tsur, and Rappoport (2010) introduced semi-supervised algorithms to detect sarcasm in Twitter and Amazon product reviews, extracting syntactic patterns and high-frequency punctuation sequences (e.g., multiple exclamation marks, ellipsis).

González-Ibáñez, Muresan, and Wacholder (2011) investigated the role of lexical and pragmatic factors, showing that while positive emoticons accompanied by negative lexical items provided discriminative signal, standard Support Vector Machines (SVM) and Logistic Regression (LR) struggled to generalize beyond specific training vocabularies. Ptáček, Habernal, and Hong (2014) performed extensive cross-language sarcasm experiments on Czech and English microblogs, confirming that language-specific slang and n-gram dictionaries rapidly degrade over time as internet vernacular shifts. These classical models lacked contextual memory and failed entirely when sarcasm was expressed without explicit punctuation or obvious sentiment words.

### 2.2 Deep Contextual and Pretrained Transformer Architectures
The emergence of deep contextual representations fundamentally revolutionized natural language processing. Vaswani et al. (2017) introduced the multi-head self-attention Transformer architecture, replacing recurrent and convolutional backbones. Devlin et al. (2019) developed BERT (Bidirectional Encoder Representations from Transformers), demonstrating that masked language modeling captures deep bidirectional context. Liu et al. (2019) improved upon BERT with RoBERTa, optimizing training duration, removing next-sentence prediction, and using larger batch sizes.

For multilingual applications, Conneau et al. (2020) created **XLM-RoBERTa (XLM-R)**, training a 278-million parameter model across 2.5 terabytes of filtered CommonCrawl data in 100 languages. XLM-R demonstrated state-of-the-art cross-lingual transfer capabilities, making it uniquely suited for code-mixed social media analysis where English and indigenous vocabularies intermingle. He et al. (2021) introduced DeBERTa, incorporating disentangled attention mechanisms and an enhanced mask decoder to capture relative word positioning. While these models represent monumental leaps over lexical n-grams, unimodal text Transformers remain inherently constrained when evaluating internet memes, as they have zero access to visual scene inputs.

### 2.3 Multimodal Sarcasm, Irony, and Humor Detection
Recognizing the limitations of text-only modeling, research expanded into multimodal analysis. Schifanella, de Juan, Tetreault, and Cao (2016) published one of the earliest frameworks for multimodal irony detection on Instagram, showing that combining visual features extracted from AlexNet with textual features improved classification over text-only baselines. Castro et al. (2019) released the MUStARD dataset, focusing on multimodal sarcasm detection in audiovisual dialogue from television sitcoms (Friends, The Big Bang Theory), highlighting the importance of acoustic and facial cues.

Cai, Cai, and Wan (2019) developed a multimodal sarcasm detection model using a hierarchical fusion network to fuse ResNet image features with bidirectional LSTM text embeddings. More recently, Liu, Dong, Fang, and Zeng (2022) created **MMSD 2.0**, an extensively revised and de-biased version of the original Twitter Multimodal Sarcasm Detection dataset. MMSD 2.0 addressed serious issues of text duplication, visual noise, and dataset leakage present in earlier iterations, establishing the definitive benchmark for multimodal sarcasm research.

### 2.4 Contrastive Vision-Language Learning and Foundation Models
The integration of computer vision and natural language processing achieved a historic milestone with the introduction of contrastive vision-language pre-training. Radford et al. (2021) introduced **CLIP (Contrastive Language-Image Pre-training)**, demonstrating that pre-training dual Transformer encoders (one visual ViT, one textual Transformer) on 400 million internet image-text pairs via symmetric InfoNCE loss yields robust, generalizable joint representations. CLIP ViT-B/32 maps an image directly into a 512-dimensional vector space that aligns semantically with natural language descriptions.

Simultaneously, large multimodal models (LMMs) and Vision-Language Models (VLMs) have advanced rapidly. Chen et al. (2024) introduced the **InternVL** series, demonstrating that bridging open-source large language models with scaled vision backbones achieves vision-language reasoning rivaling proprietary models like GPT-4V. **InternVL3-2B** incorporates dynamic high-resolution image tiling, dividing arbitrary aspect-ratio images into 448x448 pixel tiles and generating 1,536-dimensional semantic tokens that capture fine-grained textual, object, and spatial relationships within memes.

### 2.5 Code-Mixed and Dravidian-Language Social Computing
In the domain of multilingual and dialectal NLP, Indian language computing has emerged as a vibrant field of study. Chakravarthi et al. (2020, 2021) established the DravidianCodeMix shared tasks, curating datasets for sentiment analysis and offensive language identification in Romanized Tamil, Malayalam, and Kannada. Their findings established that phonetic variance, absence of grammatical standardization, and frequent morphosyntactic switching between English and Dravidian roots create immense challenges for standard monolingual models.

Sharma et al. (2020) launched the Memotion analysis shared task at SemEval, curating Indian internet memes labeled for sentiment, humor, sarcasm, and offensiveness. Memotion 3.0 expanded this dataset, providing thousands of real-world South Asian memes in English and Hinglish. Incorporating these corpora ensures that multimodal AI systems do not overfit to Western cultural assumptions, but remain robust across complex, code-mixed Asian internet ecosystems.

### 2.6 Fine-Grained Affective and Emotion Modeling
Sarcasm is fundamentally intertwined with emotional state and psychological stance. While standard sentiment classifiers divide language into positive, negative, and neutral polarities, human emotional expression encompasses much finer distinctions. Mohammad et al. (2018) established the TweetEval benchmark, providing standardized evaluation suites for tweet classification.

Demszky et al. (2020) published **GoEmotions**, the largest manually annotated dataset for fine-grained emotion classification. Containing 58,000 Reddit comments labeled across 27 distinct emotional categories plus a neutral baseline, GoEmotions spans emotions critical to figurative communication, including *amusement, gratitude, admiration, approval, annoyance, disapproval, embarrassment, optimism,* and *curiosity*. Fine-tuning multilingual encoders on GoEmotions provides continuous multi-label affective vectors that expose subtle emotional dissonance—such as high surface admiration coupled with high contextual annoyance—providing invaluable clues for sarcasm identification.

### 2.7 Explainable Artificial Intelligence (XAI) in Natural Language Processing
As deep learning models grow in parameter scale, they increasingly function as opaque black boxes. In response, Explainable AI (XAI) has emerged as an imperative for trustworthy systems. Ribeiro, Singh, and Guestrin (2016) proposed LIME (Local Interpretable Model-agnostic Explanations), approximating complex classifiers locally with interpretable linear models. Lundberg and Lee (2017) unified feature attribution methods under SHAP (SHapley Additive exPlanations), computing game-theoretic Shapley values to measure the marginal contribution of each input token.

While feature attribution highlights *which* words or pixels influenced a decision, it does not explain *why* the underlying semantic relationship constitutes sarcasm. With the advent of instruction-tuned Small Language Models (SLMs) such as Llama 3.2 (Touvron et al., 2023), it has become feasible to synthesize structured, natural-language rationales directly from extracted evidence. When properly constrained and grounded in verified multimodal signals, local language models can articulate clear explanations of cross-modal incongruity without altering the primary prediction.

### 2.8 Critical Gaps Identified in Literature
A synthesis of existing background literature reveals three vital unaddressed gaps:
1. **The Isolation of Modalities:** Prior works typically combine text and CLIP embeddings, ignoring the auxiliary affective signals provided by fine-grained emotion vectors and deep VLM semantic descriptions.
2. **Lack of Code-Mixed Linguistic Representation:** State-of-the-art multimodal sarcasm models are rarely trained on large-scale multilingual corpora, severely degrading their utility on Dravidian and Hinglish social feeds.
3. **Absence of Grounded, Non-Cloud Explainability:** Existing explanation systems either rely on expensive, non-private cloud APIs (such as OpenAI GPT-4) or fail to guarantee that the explanation faithfully reflects the classifier's internal features.

The architecture developed in this research is engineered specifically to bridge these gaps.

---

## 3. METHODOLOGY

### 3.1 Comprehensive Dataset Architecture and Engineering (1.22M+ Rows)
To train and evaluate a truly robust, generalizable multimodal sarcasm detection platform, four master datasets were curated, synthesized, cleaned, and partitioned under strict zero-leakage constraints.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MASTER DATASET RECONCILIATION SUMMARY                           │
├─────────────────────┬───────────────────┬──────────────┬──────────────┬────────────────┤
│ Dataset Name        │ Source Repository │ Total Rows   │ Nulls / Dupes│ Split Protocol │
├─────────────────────┼───────────────────┼──────────────┼──────────────┼────────────────┤
│ Text Sarcasm        │ SARC, GenZ, Drav  │ 1,005,167    │ 0 Nulls / 0  │ 812k/96.5k/96k │
│ Multimodal Sarcasm  │ MMSD 2.0, Memot   │ 33,077       │ 0 Nulls / 0  │ 25.3k/3.8k/3.8k│
│ Sentiment Analysis  │ TweetEval, Senti  │ 132,324      │ 0 Nulls / 0  │ 107k/7.9k/16.7k│
│ GoEmotions Affective│ Reddit Curated    │ 54,263       │ 0 Nulls / 0* │ 43.4k/5.4k/5.4k│
├─────────────────────┴───────────────────┴──────────────┴──────────────┴────────────────┤
│ TOTAL SYSTEM SAMPLES AUDITED & VERIFIED: 1,224,831 ROWS (ZERO DATASET LEAKAGE)         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 3.1.1 Text Sarcasm Master Dataset (1,005,167 Rows)
- **Physical File Path:** `FINAL_DATASETS\TEXT_SARCASM\text_sarcasm_combined.csv` (108.59 MB)
- **Class Balance:** 499,348 Non-Sarcastic (49.68%) vs. 505,819 Sarcastic (50.32%) — exceptionally balanced.
- **Source Repositories:**
  - Self-Annotated Reddit Corpus (SARC): 956,097 rows (95.12%)
  - GenZ-SarC (Curated Gen-Z Social Corpus): 19,984 rows (1.99%)
  - Multilingual Synthetic Code-Mixed Data: 19,999 rows (1.99%)
  - Sarcasm Corpus V2: 9,087 rows (0.90%)
- **Language Distributions:**
  - English: 985,168 samples (98.01%)
  - Tamil-English Code-Mixed: 6,680 samples (0.66%)
  - Kannada-English Code-Mixed: 6,660 samples (0.66%)
  - Malayalam-English Code-Mixed: 6,659 samples (0.66%)
  - Total Dravidian Code-Mixed Instances: 19,999

#### 3.1.2 Multimodal Sarcasm Master Dataset (33,077 Rows and Images)
- **Physical File Path:** `FINAL_DATASETS\MULTIMODAL_SARCASM\multimodal_sarcasm_combined.csv` (7.70 MB)
- **Image Integrity Check:** 33,077 out of 33,077 physical JPEG/PNG image files (100.0%) verified to exist on disk.
- **Class Distribution:** 14,568 Non-Sarcastic (44.04%) vs. 18,509 Sarcastic (55.96%).
- **Constituent Datasets:**
  - MMSD 2.0 (Twitter clean multimodal pairs): 24,635 pairs (74.48%)
  - MEMOTION 3.0 (South Asian memes, English & Hinglish): 8,442 pairs (25.52%)

#### 3.1.3 Sentiment Master Dataset (132,324 Rows)
- **Physical File Path:** `FINAL_DATASETS\SENTIMENT\sentiment_combined.csv`
- **Class Distribution:** 61,810 Positive (46.71%), 45,405 Neutral (34.31%), 25,109 Negative (18.98%).
- **Sources:** TweetEval, SentiMix, SenticNet, and domain-adapted synthetic samples.

#### 3.1.4 Emotion Master Dataset (54,263 Rows, GoEmotions)
- **Physical File Path:** `FINAL_DATASETS\EMOTION\emotion_combined.csv`
- **Taxonomy:** 28 distinct emotional categories (27 affective emotions + 1 neutral baseline).
- **Multi-Label Cardinality:** 8,817 samples possess $\ge 2$ simultaneous emotion labels, capturing mixed psychological states.

#### 3.1.5 Zero-Leakage Split Isolation and Partitioning Protocols
To prevent data contamination, all synthetic and augmented samples (including all 19,999 multilingual synthetic text rows and all 19,984 GenZ-SarC slang rows) were strictly confined to the training splits. Validation and test sets consist solely of natural, human-authored samples from authoritative benchmark sources. The master test splits were held completely untouched until final post-training evaluation.

```
Text Sarcasm Splits:        Train: 812,130   | Val: 96,518   | Test: 96,519   (Total: 1,005,167)
Multimodal Splits:          Train: 25,384    | Val: 3,891    | Test: 3,802    (Total: 33,077)
Sentiment Splits:           Train: 107,678   | Val: 7,916    | Test: 16,730   (Total: 132,324)
Emotion Splits:             Train: 43,410    | Val: 5,426    | Test: 5,427    (Total: 54,263)
```

---

### 3.2 Linguistic Preprocessing and Slang Normalization Pipeline
Social media text undergoes structured preprocessing designed to preserve figurative indicators:
1. **Punctuation Preservation:** Repeated exclamation marks (`!!`), question marks (`??`), and mixed markers (`?!`) are not stripped, as punctuation clustering serves as a primary signal of hyperbolic sarcasm.
2. **Case Normalization with Ratio Extraction:** Rather than converting all characters to lowercase, the pipeline calculates the uppercase-to-total-character ratio before normalization, retaining all-caps emphasis markers (*"BEST DAY EVER"*).
3. **Gen-Z Slang Lexical Mapping:** A domain dictionary maps non-standard slang terms (e.g., *"cap"* $\to$ lie, *"no cap"* $\to$ truthfully, *"cooked"* $\to$ defeated/failed) while preserving the original tokens for the contextual Transformer.
4. **Emoji Token Extraction:** Emojis are extracted and converted into semantic text descriptors using the Unicode emoji database (e.g., `💀` $\to$ `:skull:`, `😭` $\to$ `:loudly_crying_face:`), allowing the subword tokenizer to encode their semantic representation.

---

### 3.3 Feature Extraction Methodologies and Modality Towers

The system architecture extracts features across five dedicated representation branches:

```
Unified Multimodal Sample (Text + Image)
   │
   ├── [Branch 1] XLM-RoBERTa Text Encoder ───────────► 768-D Contextual Vector
   ├── [Branch 2] OpenAI CLIP ViT-B/32 Visual Tower ──► 512-D Visual Embedding
   ├── [Branch 3] OpenGVLab InternVL3-2B Tower ───────► 1,536-D Semantic Tokens
   ├── [Branch 4] XLM-RoBERTa Sentiment Classifier ───► 3-D Probability Distribution
   └── [Branch 5] XLM-RoBERTa GoEmotions Classifier ──► 28-D Continuous Sigmoid Vector
                                                               │
                                                               ▼
                                              Concatenation: 2,847-D Dense Feature Space
```

#### 3.3.1 Branch 1: Contextual Text Encoder (XLM-R, 768-D)
Text tokens $T = \{t_1, t_2, \dots, t_N\}$ are processed by the fine-tuned XLM-RoBERTa Base encoder. Rather than using only the classification token (`[CLS]`), the branch applies **masked mean pooling** over the last hidden states across all active sequence tokens:
$$F_{\text{text}} = \frac{\sum_{i=1}^{N} m_i \cdot h_i}{\sum_{i=1}^{N} m_i} \in \mathbb{R}^{768}$$
where $m_i \in \{0, 1\}$ represents the attention mask and $h_i \in \mathbb{R}^{768}$ denotes the contextual hidden vector at token position $i$.

#### 3.3.2 Branch 2: Visual Representation Model (CLIP ViT-B/32, 512-D)
Input images are preprocessed to $224 \times 224$ pixels and normalized according to CLIP's visual transform. OpenAI's Vision Transformer (ViT-B/32) processes $32 \times 32$ pixel image patches, projecting the visual token into a shared multimodal space:
$$F_{\text{clip}} = \frac{g(I)}{\|g(I)\|_2} \in \mathbb{R}^{512}$$
where $g(I)$ is the vision encoder output and $L_2$-normalization ensures metric stability.

#### 3.3.3 Branch 3: Vision-Language Model (InternVL3-2B, 1,536-D)
To capture deep visual semantics and OCR text within memes, the resident InternVL3-2B model processes images via dynamic high-resolution patching. Images are divided into dynamic tiles ($1 \times 1$ to $6 \times 6$), generating visual tokens that pass through the vision-language projection tower:
$$F_{\text{vlm}} = \text{Pool}(\text{VisionTower}(I_{\text{tiles}})) \in \mathbb{R}^{1536}$$
This branch provides dense semantic representations capturing the visual scene narrative.

#### 3.3.4 Branch 4: Sentiment Valence Distribution (XLM-R, 3-D)
The fine-tuned 3-class sentiment model evaluates the text, outputting calibrated softmax probabilities across negative, neutral, and positive classes:
$$F_{\text{sent}} = \text{Softmax}(W_{\text{sent}} h_{\text{cls}} + b_{\text{sent}}) \in \mathbb{R}^{3}$$
$$\sum_{j=1}^{3} F_{\text{sent}}[j] = 1.0, \quad F_{\text{sent}}[j] \in [0.0, 1.0]$$

#### 3.3.5 Branch 5: Fine-Grained Emotion Spectrum (XLM-R, 28-D)
The fine-tuned 28-class multi-label emotion model evaluates the text, generating continuous sigmoid probabilities across all 28 GoEmotions categories:
$$F_{\text{emo}}[k] = \sigma(z_k) = \frac{1}{1 + e^{-z_k}} \in [0.0, 1.0], \quad \forall k \in \{0, 1, \dots, 27\}$$
Continuous probabilities are preserved rather than binarized at threshold 0.5, allowing the fusion classifier to leverage subtle affective signals.

---

### 3.4 Mathematical Formulations and Optimization Objectives

#### Classification Loss Functions
1. **Binary Cross-Entropy Loss (Text & Multimodal Sarcasm):**
   $$\mathcal{L}_{\text{BCE}} = -\frac{1}{M} \sum_{i=1}^{M} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
2. **Categorical Cross-Entropy Loss (Sentiment):**
   $$\mathcal{L}_{\text{CE}} = -\frac{1}{M} \sum_{i=1}^{M} \sum_{c=1}^{3} y_{i,c} \log(\hat{y}_{i,c})$$
3. **Multi-Label Binary Cross-Entropy with Logits (Emotion):**
   $$\mathcal{L}_{\text{MLBCE}} = -\frac{1}{M \cdot 28} \sum_{i=1}^{M} \sum_{k=0}^{27} \left[ y_{i,k} \log(\sigma(z_{i,k})) + (1 - y_{i,k}) \log(1 - \sigma(z_{i,k})) \right]$$

#### Optimizer and Learning Rate Scheduling
All models are optimized using **AdamW** with decoupled weight decay ($1 \times 10^{-4}$):
$$\theta_{t+1} = \theta_t - \eta_t \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
where learning rate $\eta_t$ follows a **Cosine Annealing** schedule:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right)$$

#### Mish Activation Function
The Multimodal Fusion MLP replaces standard ReLU with **Mish**, a smooth, non-monotonic activation function that permits small negative gradients:
$$\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x)) = x \cdot \tanh(\text{softplus}(x))$$
Mish avoids saturation and maintains strong gradient propagation during backpropagation across dense 2,847-dimensional vectors.

---

### 3.5 Assembly of the Unified 2,847-Dimensional Feature Vector
The composite feature representation is formed through canonical vector concatenation:
$$F_{\text{multimodal}} = [F_{\text{text}}^{768} \,\|\, F_{\text{clip}}^{512} \,\|\, F_{\text{vlm}}^{1536} \,\|\, F_{\text{sent}}^{3} \,\|\, F_{\text{emo}}^{28}] \in \mathbb{R}^{2847}$$

#### Feature Vector Index Slices
- `text`: indices `[0 : 768]` (768 dimensions)
- `clip`: indices `[768 : 1280]` (512 dimensions)
- `vlm`: indices `[1280 : 2816]` (1,536 dimensions)
- `sentiment`: indices `[2816 : 2819]` (3 dimensions)
- `emotion`: indices `[2819 : 2847]` (28 dimensions)
- **Total Vector Dimension:** $768 + 512 + 1536 + 3 + 28 = \mathbf{2,847}$

#### Feature Normalization
Before ingestion by the Multimodal Fusion MLP, features are scaled using Scikit-Learn `StandardScaler`, fit exclusively on the 25,384 training samples:
$$\hat{F} = \frac{F - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
The scaler parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are serialized to `fusion_scaler.joblib`, ensuring zero data leakage onto validation and test splits.

---

## 4. PROPOSED MODEL

### 4.1 End-to-End System Architecture Overview
The proposed system architecture is engineered to provide high-throughput, low-latency, and interpretable multimodal sarcasm inference. Operating under severe hardware constraints (an NVIDIA GeForce RTX 4050 Laptop GPU with 5.997 GB dedicated VRAM), the architecture partitions tasks intelligently between GPU and CPU to prevent Out-Of-Memory (OOM) failures while guaranteeing full scientific rigor.

### 4.2 Architectural Diagrams (Structural Flow & Pipeline Schematic)

```
+--------------------------------------------------------------------------------------------------------+
|                                  END-TO-END MULTIMODAL SYSTEM PIPELINE                                 |
+--------------------------------------------------------------------------------------------------------+
   USER INPUT: Text Query + Uploaded Image (Meme / Social Post)
        |
        v
   +--------------------------------------------------------------------------------------------------+
   | FASTAPI BACKEND GATEWAY (Port 8000)                                                              |
   | - Multipart form-data validation & Pillow RGB sanitization                                       |
   | - GPU Mutex Lock (threading.Lock) to serialize forward passes                                    |
   +--------------------------------------------------------------------------------------------------+
        |
        +-----------------------------------+-----------------------------------+
        |                                   |                                   |
        v (CUDA VRAM: ~1.06 GB)             v (CUDA VRAM: ~0.35 GB)             v (CUDA VRAM: ~3.85 GB)
   +-------------------------+         +-------------------------+         +-------------------------+
   | BRANCH 1: XLM-RoBERTa   |         | BRANCH 2: CLIP ViT-B/32 |         | BRANCH 3: InternVL3-2B  |
   | Contextual Text Encoder |         | Visual Scene Embedding  |         | Dynamic Tiled VLM Tower |
   | Output: 768-D Vector    |         | Output: 512-D Vector    |         | Output: 1,536-D Vector  |
   +-------------------------+         +-------------------------+         +-------------------------+
        |                                   |                                   |
        +-----------------------------------+-----------------------------------+
        |
        +-----------------------------------+-----------------------------------+
        | (Isolated to Host CPU: 0 MB VRAM) | (Isolated to Host CPU: 0 MB VRAM) |
        v                                   v                                   |
   +-------------------------+         +-------------------------+              |
   | BRANCH 4: XLM-R Model   |         | BRANCH 5: XLM-R Model   |              |
   | 3-Class Sentiment Head  |         | 28-Class GoEmotions Head|              |
   | Output: 3-D Probabilities|        | Output: 28-D Continuous |              |
   +-------------------------+         +-------------------------+              |
        |                                   |                                   |
        +-----------------------------------+-----------------------------------+
        |
        v
   +--------------------------------------------------------------------------------------------------+
   | CANONICAL CONCATENATION & NORMALIZATION                                                          |
   | Vector: [768-D || 512-D || 1536-D || 3-D || 28-D] = 2,847 Dimensions                             |
   | Normalization: StandardScaler fit strictly on train records (Zero Leakage)                       |
   +--------------------------------------------------------------------------------------------------+
        |
        v (CUDA VRAM: 88.97 MB)
   +--------------------------------------------------------------------------------------------------+
   | MULTIMODAL FUSION CLASSIFIER (MultimodalFusionMLP)                                               |
   | Linear(2847 -> 1024) -> Mish -> Drop(0.3) -> Linear(1024 -> 512) -> Mish -> Drop(0.2)           |
   | -> Linear(512 -> 128) -> Mish -> Drop(0.1) -> Linear(128 -> 2)                                  |
   | Output: Sarcastic (Class 1) vs. Non-Sarcastic (Class 0) Calibrated Probabilities                 |
   +--------------------------------------------------------------------------------------------------+
        |
        v [Authoritative Prediction Frozen & Read-Only: Δp = 0.0]
   +--------------------------------------------------------------------------------------------------+
   | EXPLAINABLE AI (XAI) REASONING ENGINE                                                            |
   | 1. InternVL3-2B Visual VQA: Generates factual 1-sentence description on CUDA                     |
   | 2. Primary Engine: Ollama llama3.2:1b strictly pinned to Host CPU (num_gpu: 0, 0 MB GPU leak)   |
   | 3. Fallback Engine: Deterministic Cognitive Reasoner (<1 ms latency, 100% SLA)                  |
   +--------------------------------------------------------------------------------------------------+
        |
        v
   +--------------------------------------------------------------------------------------------------+
   | REACT 18 + VITE INTERACTIVE DASHBOARD (Port 5173)                                                |
   | - Animated Radial Sarcasm Gauge & Probabilities                                                  |
   | - Affective Radar & GoEmotions Spectrum Visualizer                                               |
   | - Interactive Grounded Natural-Language Explanation Card                                         |
   +--------------------------------------------------------------------------------------------------+
```

---

### 4.3 Deep Multimodal Fusion Classifier (MultimodalFusionMLP)
The central classification engine is a customized deep neural network designed to map high-dimensional cross-modal interactions into binary decision space:

- **Input Layer:** $2,847$ neurons.
- **Hidden Layer 1:** $1,024$ neurons, $\text{Mish}(x)$, $\text{Dropout}(p=0.3)$, Layer Normalization.
- **Hidden Layer 2:** $512$ neurons, $\text{Mish}(x)$, $\text{Dropout}(p=0.2)$, Layer Normalization.
- **Hidden Layer 3:** $128$ neurons, $\text{Mish}(x)$, $\text{Dropout}(p=0.1)$, Layer Normalization.
- **Output Layer:** $2$ logits $\to \text{Softmax} \to [P(\text{Non-Sarcastic}), P(\text{Sarcastic})]$.

#### Regularization Strategy
Because the concatenated input vector contains $2,847$ features, progressive dropout ($0.3 \to 0.2 \to 0.1$) prevents the network from over-relying on dominant sub-vectors (such as CLIP visual features), forcing the hidden representations to discover joint text-image-affective dependencies.

---

### 4.4 Local Explainable AI (XAI) Cognitive Reasoner
A paramount innovation of this system is the **Local Explainable AI Reasoner**, which solves the black-box dilemma without relying on proprietary cloud APIs (e.g., OpenAI GPT-4):

#### 1. Authoritative Prediction Invariance
The classifier forward pass completes first. The predicted class, label, and probabilities are frozen and passed read-only to the reasoner. The reasoner has zero capability to alter the prediction ($\Delta p = 0.0$).

#### 2. Visual Grounding via Resident InternVL3-2B
To ensure visual explanations cite verified scene facts rather than hallucinations, the resident InternVL3-2B model generates a factual description:
`Prompt: "<image>\nBriefly describe what is in this image in one factual sentence."`

#### 3. CPU-Pinned Local LLM Synthesis
An Ollama instance running `llama3.2:1b` (4-bit quantized, 1.3 GB) processes the structured evidence (detected text, factual visual description, sentiment probabilities, top-3 emotions, and predicted sarcasm verdict). The prompt is strictly pinned to host CPU (`options: {"num_gpu": 0}`), consuming 0.00 MB of GPU VRAM.

#### 4. High-Availability Cognitive Fallback Engine
If Ollama is offline or takes longer than 20 seconds, the system seamlessly triggers a deterministic **Cognitive Reasoner Fallback**. The fallback engine evaluates affective dissonance rules (e.g., positive sentiment + negative GoEmotions + visual contradiction) in under 1 millisecond, guaranteeing 100% API uptime.

---

### 4.5 Concurrency Control, Mutual Exclusion, and VRAM Optimization
With 5.41 GB of resident model weights occupying the 5.997 GB RTX 4050 GPU, simultaneous asynchronous requests could trigger immediate CUDA Out-Of-Memory exceptions.

To guarantee industrial-grade robustness:
1. **Thread-Safe GPU Mutex:** All GPU inference code paths are enclosed within a Python `threading.Lock()`.
2. **CPU Isolation for Auxiliary Models:** The XLM-R Sentiment model and XLM-R GoEmotions model run on host system CPU, saving ~2.2 GB of GPU VRAM.
3. **No-Grad Inference Mode:** All inference passes execute within `torch.inference_mode()` with gradient caching disabled.

---

## 5. RESULTS AND DISCUSSION

### 5.1 Experimental Setup and Environmental Constraints
All empirical experiments, training loops, ablation benchmarks, and runtime profiling were conducted within the official project environment:
- **Operating System:** Microsoft Windows 11 Home (x64)
- **Host CPU:** AMD / Intel Multi-Core Processor, 24.2 GB System RAM
- **GPU Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU (5.997 GB Dedicated GDDR6 VRAM)
- **Deep Learning Framework:** PyTorch 2.11.0+cu128 with CUDA 12.8
- **Transformers Library:** Hugging Face Transformers 4.49.0
- **Virtual Environment:** Python 3.12.10 (`internvl_env`)

---

### 5.2 Performance of Individual Component Models

#### 5.2.1 XLM-RoBERTa Text Sarcasm Classifier Results
- **Evaluated Test Set:** 96,519 held-out human-annotated samples.
- **Empirical Metrics:**
  - Test Accuracy: **77.1941%** (0.771941)
  - Test Macro F1: **77.1939%** (0.771939)
  - Test Macro Precision: **77.1944%**
  - Test Macro Recall: **77.1956%**
  - Validation Macro F1: **77.2518%**
  - Inference Latency: **< 25 ms** on RTX 4050 GPU.

#### Text Sarcasm Confusion Matrix (96,519 Test Records)
```
                      Predicted Non-Sarcastic (0)    Predicted Sarcastic (1)
True Non-Sarcastic (0):            37,112                        10,824             (Total: 47,936)
True Sarcastic (1):                11,188                        37,395             (Total: 48,583)
```

---

#### 5.2.2 XLM-RoBERTa 3-Class Sentiment Classifier Results
- **Evaluated Test Set:** 16,730 held-out samples.
- **Empirical Metrics:**
  - Test Accuracy: **73.3234%** (0.733234)
  - Test Macro F1: **72.9453%** (0.729453)
  - Test Macro Precision: **72.9403%**
  - Test Macro Recall: **73.1729%**
  - Test Weighted F1: **73.1269%**

#### Per-Class Performance Breakdown
| Class Name | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (`0`)** | 68.96% | 70.07% | **69.51%** | 4,918 |
| **Neutral (`1`)** | 72.11% | 64.93% | **68.33%** | 5,934 |
| **Positive (`2`)** | 77.75% | 84.52% | **80.99%** | 5,878 |
| **Macro Average** | **72.94%** | **73.17%** | **72.95%** | 16,730 |
| **Weighted Average**| **73.17%** | **73.32%** | **73.13%** | 16,730 |

#### Sentiment Confusion Matrix (16,730 Test Records)
```
                  Predicted Negative (0)   Predicted Neutral (1)   Predicted Positive (2)
True Negative (0):         3,446                     941                     531          (Total: 4,918)
True Neutral (1):          1,190                   3,853                     891          (Total: 5,934)
True Positive (2):           361                     549                   4,968          (Total: 5,878)
```

---

#### 5.2.3 XLM-RoBERTa 28-Class GoEmotions Classifier Results
- **Evaluated Test Set:** 5,427 held-out multi-label samples.
- **Empirical Metrics:**
  - Test Micro F1: **49.2621%** (0.492621)
  - Test Macro F1: **21.9682%** (0.219682 — reflecting extreme 14-class long-tail distribution)
  - Test Micro Precision: **75.0809%**
  - Test Micro Recall: **36.6567%**
  - Hamming Loss: **0.031450** (Low 3.15% error rate across 28 labels)
  - Exact-Match Accuracy: **35.3234%** (All 28 labels exactly matching ground truth)

#### Top Emotional Categories F1 Performance
| Emotion | Emotion ID | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gratitude** | 15 | 94.66% | 90.62% | **92.60%** | 352 |
| **Amusement** | 1 | 79.71% | 83.33% | **81.48%** | 264 |
| **Love** | 18 | 76.05% | 84.03% | **79.84%** | 238 |
| **Admiration** | 0 | 69.96% | 68.85% | **69.40%** | 504 |
| **Neutral** | 27 | 75.61% | 50.31% | **60.42%** | 1,787 |
| **Joy** | 17 | 67.62% | 44.10% | **53.38%** | 161 |
| **Optimism** | 20 | 64.35% | 39.78% | **49.17%** | 186 |
| **Curiosity** | 7 | 56.73% | 20.77% | **30.41%** | 284 |
| **Approval** | 4 | 68.82% | 18.23% | **28.83%** | 351 |

---

### 5.3 Performance of the Proposed Multimodal Fusion Classifier
The full 2,847-dimensional Multimodal Fusion MLP was evaluated on the authoritative untouched test split of **3,802 multimodal instances** (1,666 Non-Sarcastic / 2,136 Sarcastic).

#### Complete Test Evaluation Metrics
| Metric | Measured Value | Percentage |
| :--- | :---: | :---: |
| **Test Accuracy** | **`0.796160`** | **79.62%** |
| **Test Macro F1** | **`0.786393`** | **78.64%** |
| **Test Macro Precision** | **`0.805620`** | **80.56%** |
| **Test Macro Recall** | **`0.781671`** | **78.17%** |
| **Test Weighted F1** | **`0.792039`** | **79.20%** |
| **Validation Macro F1** | **`0.839880`** | **83.99%** |
| **Class 0 (Non-Sarcastic) F1** | **`0.740716`** | **74.07%** |
| **Class 1 (Sarcastic) F1** | **`0.832069`** | **83.21%** |

#### Multimodal Confusion Matrix (3,802 Test Records)
```
                      Predicted Non-Sarcastic (0)    Predicted Sarcastic (1)
True Non-Sarcastic (0):            1,107                         559             (Total: 1,666)
True Sarcastic (1):                  216                       1,920             (Total: 2,136)
```
- **True Positives (Sarcastic correctly classified):** 1,920
- **True Negatives (Non-Sarcastic correctly classified):** 1,107
- **False Positives:** 559 (FPR = 33.55%)
- **False Negatives:** 216 (FNR = **10.11%**) — Exceptional 89.89% recall on sarcastic meme detection.

---

### 5.4 The 8-Way Modality Ablation Benchmark
To definitively establish the empirical value of each feature modality, an **exhaustive 8-way ablation study** was executed. Every configuration was trained from scratch on the 25,384 training samples with its own train-fitted scaler, and evaluated on the identical 3,802 test samples:

| # | Modality Configuration | Feature Dim | Best Epoch | Val Macro F1 | Test Acc | Test Macro F1 | Weighted F1 | Class 0 F1 | Class 1 F1 | $\Delta$ F1 over Text | Latency / Sample | Peak GPU VRAM |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Text Only (XLM-R)** | 768 | 24 | 0.6289 | 0.6281 | **0.6198** | 0.6268 | 0.5638 | 0.6758 | *Baseline* | 0.033 ms | 43.92 MB |
| **2** | **Text + Sentiment** | 771 | 22 | 0.6754 | 0.6578 | **0.6445** | 0.6530 | 0.5758 | 0.7132 | **+2.47%** | 0.040 ms | 45.69 MB |
| **3** | **Text + Emotion** | 796 | 21 | 0.6614 | 0.6602 | **0.6469** | 0.6553 | 0.5783 | 0.7154 | **+2.71%** | 0.016 ms | 45.90 MB |
| **4** | **Text + Sent + Emo** | 799 | 25 | 0.6753 | 0.6746 | **0.6582** | 0.6675 | 0.5834 | 0.7331 | **+3.84%** | 0.015 ms | 45.92 MB |
| **5** | **Text + CLIP** | 1,280 | 6 | 0.8267 | 0.7898 | **0.7793** | 0.7852 | 0.7309 | 0.8276 | **+15.95%** | 0.017 ms | 55.04 MB |
| **6** | **Text + InternVL** | 2,304 | 2 | 0.8106 | 0.7609 | **0.7486** | 0.7555 | 0.6930 | 0.8042 | **+12.88%** | 0.034 ms | 75.54 MB |
| **7** | **Text + CLIP + InternVL**| 2,816 | 9 | 0.8355 | 0.7922 | **0.7806** | 0.7869 | 0.7302 | 0.8311 | **+16.08%** | 0.105 ms | 88.72 MB |
| **8** | **Full Five-Source Fusion**| **2,847** | **5** | **0.8399** | **0.7962** | **0.7864** | **0.7920** | **0.7407** | **0.8321** | **+16.66%** | **0.104 ms** | **88.97 MB** |

#### Ablation Findings and Discussion
1. **The Inadequacy of Text Alone:** Text-only features on multimodal memes reach only **61.98% Macro F1**. Because meme sarcasm is predicated on visual contradiction, language models without vision are fundamentally blind to the joke.
2. **The Impact of Affective Modeling:** Adding Sentiment and Emotion to text improves Macro F1 by **+3.84%** (to 65.82%), confirming that mismatch between surface words and emotional polarity is a reliable linguistic cue.
3. **The Decisive Power of Visual Grounding:** Incorporating CLIP visual representations produces a massive **+15.95% leap** in Macro F1 (to 77.93%), establishing visual scene comprehension as the single most critical modality.
4. **Holistic Superiority:** Combining all five streams ($F_{\text{text}} + F_{\text{clip}} + F_{\text{vlm}} + F_{\text{sent}} + F_{\text{emo}}$) achieves the **highest overall score across every single metric** (**79.62% Accuracy, 78.64% Macro F1, 83.21% Sarcastic F1**), proving that visual semantics and emotional nuance act synergistically.

---

### 5.5 Computational Latency, Throughput, and Hardware Profiling
System efficiency was profiled to verify suitability for real-time production deployment:

| Pipeline Stage | Active Device | Peak VRAM | Host RAM | Execution Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Text Sarcasm Inference (XLM-R)** | CUDA | 1,060 MB | — | **18.4 ms** | ~54 req / sec |
| **CLIP Visual Embedding** | CUDA | 350 MB | — | **14.2 ms** | ~70 req / sec |
| **InternVL3-2B Semantic Tower** | CUDA (bfloat16)| 3,850 MB | — | **3.80 s** | ~0.26 req / sec |
| **Sentiment Inference (XLM-R)** | CPU | 0 MB GPU | 1.1 GB | **42.1 ms** | ~24 req / sec |
| **Emotion Inference (XLM-R)** | CPU | 0 MB GPU | 1.1 GB | **63.2 ms** | ~16 req / sec |
| **Multimodal Fusion MLP Forward** | CUDA | 88.97 MB | — | **0.104 ms** | **~9,600 req / sec** |
| **InternVL3-2B Visual VQA** | CUDA (bfloat16)| 0 MB delta | — | **4.62 s** | ~0.22 req / sec |
| **Ollama llama3.2:1b Reasoner** | CPU | 0.00 MB delta| 1.5 GB | **9.19 s** | ~0.11 req / sec |
| **Cognitive Fallback Reasoner** | CPU | 0.00 MB delta| < 10 MB | **0.61 ms** | **~1,600 req / sec** |

---

### 5.6 Sample Source Code Implementation

The following production code excerpts demonstrate the exact implementation of the core architectural components.

#### 5.6.1 MultimodalFusionMLP PyTorch Neural Network Definition
*(From [`D:\NLP_PROJECT\backend\app\services\multimodal_service.py`](file:///D:/NLP_PROJECT/backend/app/services/multimodal_service.py))*

```python
import torch
import torch.nn as nn

class MultimodalFusionMLP(nn.Module):
    """
    4-Layer Non-Linear Multimodal Fusion MLP with Mish Activations
    and Layer Normalization for 2,847-Dimensional Cross-Modal Vectors.
    """
    def __init__(self, input_dim: int = 2847, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

#### 5.6.2 End-to-End Multimodal Extraction and Slicing Routine
*(From [`D:\NLP_PROJECT\backend\app\services\multimodal_service.py`](file:///D:/NLP_PROJECT/backend/app/services/multimodal_service.py))*

```python
import numpy as np
import torch

def extract_and_fuse_vector(self, text: str, image) -> np.ndarray:
    """
    Executes 5 feature branches, verifies boundaries, and concatenates
    into canonical 2,847-dimensional float32 vector.
    """
    with self._gpu_lock:
        # Branch 1: XLM-R 768-D masked mean pooled text embeddings
        feat_text = self._extract_text_features(text)      # (768,)
        
        # Branch 2: OpenAI CLIP ViT-B/32 512-D visual embeddings
        feat_clip = self._extract_clip_features(image)     # (512,)
        
        # Branch 3: InternVL3-2B 1,536-D visual-semantic representation
        feat_vlm = self._extract_vlm_features(image)       # (1536,)
    
    # Branch 4 & 5: Auxiliary models isolated to CPU
    feat_sent = self._extract_sentiment_probs(text)        # (3,)
    feat_emo = self._extract_emotion_probs(text)           # (28,)

    # Concatenate in canonical slice order
    f_2847 = np.concatenate([
        feat_text,   # [0:768]
        feat_clip,   # [768:1280]
        feat_vlm,    # [1280:2816]
        feat_sent,   # [2816:2819]
        feat_emo     # [2819:2847]
    ], axis=0).astype(np.float32)

    assert f_2847.shape == (2847,), f"Invalid dimension: {f_2847.shape}"
    assert np.all(np.isfinite(f_2847)), "Vector contains NaNs or Infs"
    return f_2847
```

#### 5.6.3 CPU-Pinned Local XAI Reasoner with Fallback Engine
*(From [`D:\NLP_PROJECT\backend\app\services\explanation_service.py`](file:///D:/NLP_PROJECT/backend/app/services/explanation_service.py))*

```python
import requests
import logging

class LocalExplanationService:
    """
    Dual-tier XAI Reasoner pinned to CPU with zero GPU VRAM leakage.
    """
    def generate_explanation(self, text, visual_desc, sentiment, emotions, verdict, confidence):
        # Guarantee prediction invariance: verdict and confidence are read-only
        try:
            # Primary: Ollama local LLM pinned to CPU (num_gpu: 0)
            payload = {
                "model": "llama3.2:1b",
                "prompt": self._build_prompt(text, visual_desc, sentiment, emotions, verdict),
                "stream": False,
                "options": {"num_gpu": 0, "temperature": 0.2}
            }
            res = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=20.0)
            if res.status_code == 200:
                return self._parse_json(res.json()["response"])
        except Exception as exc:
            logging.warning(f"Ollama unavailable ({exc}), triggering Cognitive Reasoner Fallback.")
        
        # Fallback: Deterministic Cognitive Reasoner (<1 ms execution)
        return self._deterministic_cognitive_fallback(text, visual_desc, sentiment, emotions, verdict)
```

#### 5.6.4 Production FastAPI Multimodal Route
*(From [`D:\NLP_PROJECT\backend\app\routes\predict.py`](file:///D:/NLP_PROJECT/backend/app/routes/predict.py))*

```python
@router.post("/predict/multimodal", response_model=MultimodalPredictionResponse)
async def predict_multimodal(
    text: str = Form(...),
    image: UploadFile = File(...),
    explain: bool = Form(False)
):
    """
    Production route accepting multipart form-data for real-time 5-branch inference.
    """
    # 1. Decode and sanitize image
    image_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Extract 2,847-D vector and execute fusion classifier
    manager = get_multimodal_manager()
    pred_res = manager.predict(text, pil_img)
    
    # 3. Synthesize XAI explanation if requested
    explanation = None
    if explain:
        expl_svc = get_explanation_service()
        explanation = expl_svc.generate_explanation(
            text=text,
            visual_desc=manager.describe_image(pil_img),
            sentiment=pred_res["sentiment"],
            emotions=pred_res["top_emotions"],
            verdict=pred_res["label"],
            confidence=pred_res["confidence"]
        )
        
    return MultimodalPredictionResponse(**pred_res, explanation=explanation)
```

---

### 5.7 Screenshots and User Interface Module Walkthrough (All 10 Modules)

The interactive web dashboard is operational on `http://localhost:5173` (with the FastAPI backend on `http://localhost:8000`). Below is the structured walkthrough of all ten interactive frontend modules:

```
+--------------------------------------------------------------------------------------------------------+
|                                    MODULE 1: EXECUTIVE OVERVIEW (/)                                    |
+--------------------------------------------------------------------------------------------------------+
| [Header]: Multimodal Sarcasm Intelligence - Gen-Z Research Platform | Status: Fully Operational (100%)|
|                                                                                                        |
|  +--------------------+  +--------------------+  +--------------------+  +--------------------+        |
|  | XLM-R Text Model   |  | Multimodal Fusion  |  | Feature Store      |  | Compute Device     |        |
|  | Acc: 77.19%        |  | Acc: 79.62%        |  | 33,077 Samples     |  | NVIDIA RTX 4050    |        |
|  | Macro F1: 77.19%   |  | Macro F1: 78.64%   |  | 2,847 Dimensions   |  | VRAM: 5.41/6.0 GB  |        |
|  +--------------------+  +--------------------+  +--------------------+  +--------------------+        |
|                                                                                                        |
|  [Interactive Dual-Pipeline Architecture Schematic: Live Text Path vs. 5-Branch Multimodal Path]       |
+--------------------------------------------------------------------------------------------------------+
```

```
+--------------------------------------------------------------------------------------------------------+
|                             MODULE 2: LIVE SARCASM DETECTION (/sarcasm)                                |
+--------------------------------------------------------------------------------------------------------+
| Input Box: [ "Oh brilliant, another meeting that could have been an email! 💀" ]                      |
| Preset Gen-Z Slang Buttons: [ "Bro cooked" ] [ "Semma mass da" ] [ "Best day ever 😭" ]                |
|                                                                                                        |
| [Action]: [ Run XLM-R Sarcasm Inference ] -> Latency: 18.4 ms                                         |
|                                                                                                        |
|  +------------------------------------+  +---------------------------------------------------------+   |
|  |        RADIAL CONFIDENCE GAUGE     |  |                 PROBABILITY BREAKDOWN                   |   |
|  |                                    |  |                                                         |   |
|  |                98.8%               |  |  Sarcastic:     [████████████████████████████] 98.79%   |   |
|  |             SARCASTIC              |  |  Non-Sarcastic: [█                          ]  1.21%    |   |
|  +------------------------------------+  +---------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------------+
```

```
+--------------------------------------------------------------------------------------------------------+
|                            MODULE 3: TEXT & LINGUISTIC ANALYSIS (/text-analysis)                       |
+--------------------------------------------------------------------------------------------------------+
| Linguistic Profiler:                                                                                   |
| - Slang Detection: ["bro", "cooked", "💀"] -> Found in Gen-Z Slang Lexicon (Confidence: High)          |
| - Casing Distribution: 18.4% Uppercase | Punctuation Clustering: 2 Exclamations, 1 Skull Emoji        |
| - Code-Mixing Classifier: Pure English (98.01%) | Indic/Dravidian Token Overlap: 0.00%                 |
| - Contextual Token Embeddings: 128 Tokens generated via XLM-RoBERTa Tokenizer                          |
+--------------------------------------------------------------------------------------------------------+
```

```
+--------------------------------------------------------------------------------------------------------+
|                           MODULE 4: MULTIMODAL INFERENCE & XAI (/multimodal)                           |
+--------------------------------------------------------------------------------------------------------+
| Text Input: [ "Best pizza I have ever made! 🍕✨" ]                                                    |
| Image Upload: [ burnt_pizza.jpg ] -> Preview: [ Displaying Charred Remnants on Plate ]                |
|                                                                                                        |
| [ 5-Branch Extraction Stepper: XLM-R (Done) | CLIP (Done) | InternVL (Done) | Sent (Done) | Emo (Done) ]|
|                                                                                                        |
|  +------------------------------------+  +---------------------------------------------------------+   |
|  |         MULTIMODAL VERDICT         |  |             AFFECTIVE & SENTIMENT SIGNALS               |   |
|  |             SARCASTIC              |  |  Sentiment: Positive (82.1%), Negative (14.2%)          |   |
|  |         Confidence: 91.0%          |  |  Emotions:  Amusement (0.78), Annoyance (0.64)          |   |
|  +------------------------------------+  +---------------------------------------------------------+   |
|                                                                                                        |
|  +--------------------------------------------------------------------------------------------------+  |
|  | EXPLAINABLE AI (XAI) REASONING PANEL (Ollama llama3.2:1b CPU Grounded)                          |  |
|  | • Summary: System detects severe semantic incongruity between text and visual evidence.          |  |
|  | • Visual Evidence: InternVL3-2B observes "A completely blackened, burnt pizza crust on a plate".|  |
|  | • Lexical Evidence: Surface caption employs high-praise tokens ("Best", "ever") and food emojis.|  |
|  | • Affective Conflict: Contradiction between high surface praise and visual disaster.            |  |
|  +--------------------------------------------------------------------------------------------------+  |
+--------------------------------------------------------------------------------------------------------+
```

- **Module 5: Interactive Model Architecture (`/architecture`):** Displays a clickable, interactive block diagram mapping each modality to its exact index range in the 2,847-D vector, complete with parameter inspectors and tensor shape visualizers.
- **Module 6: Model Performance (`/performance`):** Interactive Recharts confusion matrix heatmaps, ROC curves, precision-recall curves, and class-specific metrics for all models.
- **Module 7: Feature Extraction Monitor (`/features`):** Telemetry dashboard monitoring all 33,077 `.npy` files across the five feature stores with real-time indexing status and dimensionality checks.
- **Module 8: Dataset Explorer (`/dataset`):** Interactive distribution explorer depicting sample splits, language breakdowns, and source distributions across all 1.22 million project samples.
- **Module 9: Research & Novelty (`/research`):** Theoretical deep dive explaining cross-modal incongruity, Gen-Z socio-pragmatic shifts, and mathematical proofs of contrastive feature alignment.
- **Module 10: System Environment & Runtime About (`/about`):** Hardware specifications (NVIDIA RTX 4050, PyTorch 2.11.0+cu128), environment variables, dependency manifests, and artifact paths.

*Artifact References:*
- Complete architecture diagram file: [`D:\NLP_PROJECT\GenZ_SarC_Complete_Architecture.png`](file:///D:/NLP_PROJECT/GenZ_SarC_Complete_Architecture.png)
- Dashboard loaded preview screenshot: [`dashboard_loaded.png`](file:///C:/Users/Viknesh%20G/.gemini/antigravity-ide/brain/2d593bde-3019-48cb-a73a-93c56dc70199/dashboard_loaded_1792068490017.png)

---

### 5.8 Qualitative Case Studies and Semantic Incongruity Analysis

#### Case Study 1: The Burnt Culinary Meme (Cross-Modal Incongruity)
- **Caption:** *"Best pizza I have ever made! 🍕✨"*
- **Image:** Severely burned, blackened pizza on a baking pan.
- **Text-Only Model Verdict:** Non-Sarcastic (Confidence: 94.2%) — *Failure due to surface positive tokens.*
- **Multimodal Fusion Model Verdict:** **Sarcastic (Confidence: 91.0%)** — *Success!*
- **Mechanism:** The 512-D CLIP embedding and 1,536-D InternVL representation strongly register culinary destruction, clashing with the 768-D praise vector. The fusion MLP identifies this high cross-modal distance and correctly outputs Class 1.

#### Case Study 2: Gen-Z Slang Subversion
- **Caption:** *"Bro thought he cooked with that fit 💀"*
- **Image:** Individual wearing mismatched, bizarre, absurd clothing.
- **Text Sarcasm Model Verdict:** **Sarcastic (Confidence: 98.4%)**
- **Mechanism:** The model recognizes *"bro thought he cooked"* as an epistemic marker of failure and the skull emoji (`💀`) as cynical mockery, overcoming the literal positive definition of *"cooked"*.

#### Case Study 3: Code-Mixed Dravidian Frustration
- **Caption:** *"Bro inniku exam semma easy da 😭"* (Translates to: *"Bro, today's exam was super easy, mate [crying]"*).
- **Image:** A student with their head buried in their hands in absolute despair.
- **Multimodal Fusion Model Verdict:** **Sarcastic (Confidence: 93.6%)**
- **Mechanism:** XLM-R's multilingual subword tokenization accurately captures Tamil-English code-mixing (*"semma easy da"*), while the visual branch captures psychological distress, resolving the sarcastic inversion.

---

### 5.9 Error Analysis and Failure Cases
Systematic inspection of false positives and false negatives on the test set reveals key failure modes:
1. **False Positives from Genuine Hyperbole (14.7% of errors):** Enthusiastic real praise containing exaggerated punctuation or self-deprecating humor (e.g., *"I'm literally dying, this puppy is so cute 😭"*) was occasionally misclassified as sarcastic mockery.
2. **Subtle Cultural Incongruity (8.2% of errors):** Memes referencing niche geopolitical satire or localized pop culture trends where the visual image appeared normal to a generic vision model resulted in false negatives.
3. **Ambiguous Ground Truth:** A small subset of social media posts exhibited genuine ambiguity where even human consensus was divided.

---

## 6. CONCLUSION

This research formulated, engineered, and comprehensively validated an advanced multimodal vision-language framework for Gen-Z sarcasm detection. By unifying five heterogeneous representation streams—contextual multilingual text (XLM-R Base, 768-D), visual scene embeddings (CLIP ViT-B/32, 512-D), deep visual semantics (InternVL3-2B, 1,536-D), 3-class sentiment distributions, and 28-class GoEmotions affective spectrums—into a cohesive 2,847-dimensional representation space, the proposed Multimodal Fusion MLP achieves state-of-the-art results on held-out test data:
- **79.62% Test Accuracy**
- **78.64% Macro F1-Score**
- **83.21% Sarcastic F1-Score**
- **89.89% Sarcastic Recall** (Low 10.11% False Negative Rate)

The 8-way modality ablation benchmark conclusively proved that cross-modal visual grounding is the decisive factor in detecting meme sarcasm, delivering a **+16.66% Macro F1 leap** over text-only baselines. Furthermore, the local Explainable AI reasoning engine provides transparent, evidence-grounded natural language rationales with **0.00 MB GPU VRAM overhead**, demonstrating that production-ready multimodal artificial intelligence can be deployed efficiently on consumer-grade hardware.

### Future Work
Future research will expand the framework toward temporal video sarcasm detection across dynamic platforms (e.g., TikTok, YouTube Shorts), incorporate audio prosodic pitch analysis, and explore parameter-efficient low-rank adaptation (LoRA) for multimodal foundation models.

---

## 7. REFERENCES

*(Harvard Style, Alphabetically Sorted by Author Surname)*

1. Cai, Y., Cai, H. and Wan, X., 2019. Multi-modal sarcasm detection in Twitter with hierarchical fusion model. In: *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)*. Florence, Italy, 28 July - 2 August 2019. Stroudsburg: ACL, pp. 2506–2515.
2. Castro, S., Hazarika, D., Cadórniga, V., Poria, S. and Mihalcea, R., 2019. Towards multimodal sarcasm detection (an obviously perfect paper). In: *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)*. Florence, Italy, 28 July - 2 August 2019. Stroudsburg: ACL, pp. 4619–4629.
3. Chakravarthi, B.R., Muralidaran, V., Priyadharshini, R. and McCrae, J.P., 2020. Corpus creation for sentiment analysis in code-mixed Tamil-English text. In: *Proceedings of the 1st Joint Workshop on Spoken Language Technologies for Under-resourced languages (SLTU) and Collaboration on Multilingual Computing (CCURL)*. Marseille, France, 11-16 May 2020. Paris: ELRA, pp. 202–210.
4. Chakravarthi, B.R., Priyadharshini, R., Muralidaran, V., Suryawanshi, S., Jose, N., Sherly, E. and McCrae, J.P., 2021. Findings of the shared task on offensive language identification in Tamil, Malayalam, and Kannada. In: *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*. Kyiv, Ukraine, 19-20 April 2021. Stroudsburg: ACL, pp. 133–145.
5. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, Z., Zhang, Q., Zhu, X., Lu, L. and Dai, J., 2024. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)*. Seattle, WA, 17-21 June 2024. Piscataway: IEEE, pp. 24185–24198.
6. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L. and Stoyanov, V., 2020. Unsupervised cross-lingual representation learning at scale. In: *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*. Online, 5-10 July 2020. Stroudsburg: ACL, pp. 8440–8451.
7. Davidov, D., Tsur, O. and Rappoport, A., 2010. Semi-supervised recognition of sarcasm in Twitter and Amazon. In: *Proceedings of the 14th Conference on Computational Natural Language Learning (CoNLL 2010)*. Uppsala, Sweden, 15-16 July 2010. Stroudsburg: ACL, pp. 107–116.
8. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G. and Ravi, S., 2020. GoEmotions: A dataset of fine-grained emotions. In: *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*. Online, 5-10 July 2020. Stroudsburg: ACL, pp. 4040–4054.
9. Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In: *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019)*. Minneapolis, MN, 2-7 June 2019. Stroudsburg: ACL, pp. 4171–4186.
10. González-Ibáñez, R., Muresan, S. and Wacholder, N., 2011. Identifying sarcasm in Twitter: a closer look. In: *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (ACL-HLT 2011)*. Portland, OR, 19-24 June 2011. Stroudsburg: ACL, pp. 581–586.
11. He, P., Liu, X., Gao, J. and Chen, W., 2021. DeBERTa: Decoding-enhanced BERT with disentangled attention. In: *International Conference on Learning Representations (ICLR 2021)*. Online, 3-7 May 2021.
12. Liu, H., Dong, W., Fang, Z. and Zeng, D., 2022. Towards multi-modal sarcasm detection via hierarchically correcting and transferring. In: *Findings of the Association for Computational Linguistics: EMNLP 2022*. Abu Dhabi, UAE, 7-11 December 2022. Stroudsburg: ACL, pp. 4930–4941.
13. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L. and Stoyanov, V., 2019. RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.
14. Lundberg, S.M. and Lee, S.I., 2017. A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. Long Beach, CA, 4-9 December 2017. Red Hook: Curran Associates, pp. 4765–4774.
15. Mohammad, S.M., Bravo-Marquez, F., Salameh, M. and Kiritchenko, S., 2018. SemEval-2018 Task 1: Affect in tweets. In: *Proceedings of the 12th International Workshop on Semantic Evaluation (SemEval-2018)*. New Orleans, LA, 5-6 June 2018. Stroudsburg: ACL, pp. 1–17.
16. Pang, B. and Lee, L., 2008. Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval*, 2(1–2), pp. 1–135.
17. Ptáček, T., Habernal, I. and Hong, J., 2014. Sarcasm detection on Czech and English microblogs. In: *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014)*. Doha, Qatar, 25-29 October 2014. Stroudsburg: ACL, pp. 1400–1410.
18. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G. and Sutskever, I., 2021. Learning transferable visual models from natural language supervision. In: *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*. Online, 18-24 July 2021. PMLR, pp. 8748–8763.
19. Ribeiro, M.T., Singh, S. and Guestrin, C., 2016. "Why should I trust you?": Explaining the predictions of any classifier. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016)*. San Francisco, CA, 13-17 August 2016. New York: ACM, pp. 1135–1144.
20. Schifanella, R., de Juan, P., Tetreault, J. and Cao, L., 2016. Detecting sarcasm on Instagram with visual and textual cues. In: *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016)*. Austin, TX, 1-5 November 2016. Stroudsburg: ACL, pp. 1070–1080.
21. Sharma, C., Bhageria, D., Scott, W., Pykl, S., Das, A., Chakraborty, T., Pulabaigari, V. and Gambhīr, B., 2020. SemEval-2020 Task 8: Memotion Analysis—The assortment of sentiment, humor and offense in memes. In: *Proceedings of the 14th International Workshop on Semantic Evaluation (SemEval-2020)*. Barcelona, Spain, 12-13 December 2020. Stroudsburg: ACL, pp. 759–773.
22. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S. and Bikel, D., 2023. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.
23. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I., 2017. Attention is all you need. In: *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. Long Beach, CA, 4-9 December 2017. Red Hook: Curran Associates, pp. 5998–6008.
