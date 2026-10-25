# Phase 1 License and Provenance Audit Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Date of Audit:** October 21, 2026  
**Auditor:** Research-Data Auditor & Compliance Engineer  

---

## 1. Executive Summary

A thorough audit of licensing terms, intellectual property constraints, and asset redistribution boundaries was performed for all source datasets and models in Phase 1.

| Dataset | Code License | Dataset / Text License | Media / Asset License | Research Use Permitted | Redistribution Permitted | Public Commercial Use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MMSD / MMSD2.0** | MIT | Academic Research | Twitter/X TOS Apply | **Yes** | **No** (Raw images restricted) | **No** |
| **MMSD3.0** | MIT | Academic Research | Amazon Review terms / X TOS | **Yes** | **No** (Links only) | **No** |
| **SarcNet** | MIT | MIT License | Social Media Fair Research | **Yes** | **Yes** (Metadata/Code) | **Restricted** |
| **GoEmotions** | Apache 2.0 | Apache 2.0 | Reddit API Terms apply to text | **Yes** | **Yes** (With Apache attribution) | **Yes** (Subject to Apache 2.0) |
| **Civil Comments** | Apache 2.0 | CC0-1.0 (Public Domain) | N/A (Public Domain) | **Yes** | **Yes** (Unrestricted) | **Yes** (Unrestricted) |
| **MUStARD** | MIT | MIT License | Fair Use (TV Sitcom Clips) | **Yes** | **No** (Raw video redistribution) | **No** |
| **iSarcasmEval** | MIT | Academic Research | Twitter/X TOS Apply | **Yes** | **No** (Requires hydration) | **No** |
| **Synthetic Pilot V1** | MIT | CC-BY-4.0 | N/A (Textual Descriptions) | **Yes** | **Yes** (With Attribution) | **Yes** |

---

## 2. Granular Legal and Provenance Findings

### A. MMSD / MMSD2.0 / MMSD3.0
- **Upstream Repository:** `https://github.com/ZHCMOONWIND/MMSD3.0`
- **Copyright Holders:** Research authors (Cai et al. 2019, Ying et al. 2023, ZHCMOONWIND et al. 2026).
- **Text & Label License:** Intended exclusively for academic non-commercial research on multimodal NLP.
- **Image Assets:** The 24,635 images were scraped from public Twitter posts between 2018 and 2019. Under the Twitter Developer Agreement and Policy:
  - Raw images may not be redistributed in a public commercial repository without user consent or an active Developer Agreement.
  - Research hosting on internal or secured academic servers for benchmarking and model training is standard academic practice.
  - **Compliance Directive:** Keep all image files inside local `data/images/mmsd` and `.gitignore`. Do not push raw images to public Git repositories.

### B. SarcNet
- **Upstream Repository:** `https://github.com/yuetanbupt/SarcNet`
- **Copyright Holder:** Tan Yue (Beijing University of Posts and Telecommunications / NTU), 2024.
- **License Type:** Full MIT License explicitly included in upstream repository:
  > *"Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files..."*
- **Media Terms:** Image-text pairs collected from Weibo and Twitter for multilingual research. Distributed openly via Google Drive link under MIT license terms.

### C. GoEmotions
- **Upstream Repository:** `https://github.com/google-research/google-research/tree/master/goemotions`
- **Copyright Holder:** Google LLC (2020).
- **License Type:** Apache License 2.0.
- **Compliance Directive:** Fully permissive for academic and commercial research, model training, and derivative distribution. Attribution notices must accompany any derivative dataset distribution.

### D. Civil Comments
- **Upstream Source:** `https://huggingface.co/datasets/google/civil_comments` / TensorFlow Datasets
- **Origin:** Archival release from the Civil Comments commenting platform (2015-2017), curated by Jigsaw and Google.
- **License Type:** **Creative Commons Zero v1.0 Universal (CC0-1.0)** — Public Domain Dedication.
- **Compliance Directive:** Completely free of copyright restrictions. Allows unrestricted copying, modification, distribution, and research application.

### E. MUStARD
- **Upstream Repository:** `https://github.com/soujanyaporia/MUStARD`
- **Copyright Holder:** Multimodal Language Understanding Group (MLUG), 2019.
- **License Type:** MIT License for metadata, transcript text, dialogue context, and code.
- **Video Media:** Video segments originate from commercial television broadcasts (*Friends*, *The Big Bang Theory*). These are held under research Fair Use (17 U.S.C. § 107) for transformative linguistic and multimodal sarcasm detection.
- **Compliance Directive:** Metadata and transcripts may be redistributed. Raw video files must not be repackaged into commercial bundles.

### F. iSarcasm / iSarcasmEval
- **Upstream Source:** SemEval-2022 Task 6 / `headacheboy/data-iSarcasmEval`
- **Distribution Method:** Tweet IDs and binary/multiclass sarcasm labels.
- **Compliance Directive:** Due to Twitter redistribution terms, the text is not directly redistributed in the archive. Using tweet IDs requires an API hydration pipeline, which currently requires an approved Twitter/X Developer API key.

---

## 3. Compliance and Ethical Safeguards

1. **Separation of Raw and Derived Assets:** Raw data downloaded from external sources is preserved in immutable read-only form under `data/raw/`. Derived and cleaned datasets in `data/interim/` and `data/manifests/` retain original source IDs and references.
2. **Privacy and Sensitive Attributes:** No PII or private communications were scraped. All sources represent publicly indexed benchmarks or public domain archives.
3. **Synthetic Data Provenance:** Synthetic records are generated strictly from original handcrafted scenario designs and are unambiguously flagged as `SYNTHETIC_UNVERIFIED` to prevent contaminating real-world benchmark evaluations.
4. **Git Protection:** `.gitignore` enforces exclusion of large binary assets, images, and model weights to protect both repository performance and third-party copyright.
