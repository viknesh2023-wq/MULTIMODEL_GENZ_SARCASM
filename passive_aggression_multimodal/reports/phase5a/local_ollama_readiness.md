# Phase 5A Local Ollama Discovery and Environment Readiness Report

**Project Title:** Context-Aware Multimodal Passive-Aggression Detection in Social Media Using Image–Comment Fusion  
**Artifact:** `reports/phase5a/local_ollama_readiness.md`  
**Date:** October 21, 2026  
**Auditor:** Senior ML Research Engineer, Local LLM Infrastructure Engineer & Experimental Auditor  
**Milestone:** Phase 5A — Complete Local LLM Annotation Automation  
**Readiness Verdict:** `READY_FOR_LOCAL_INFERENCE`  

---

## 1. Executive Summary

A comprehensive automated inspection of the local Windows host and Python environment was conducted to discover local LLM capabilities.

**Key Findings:**
1. **Ollama Installation:** Found on system PATH at `C:\Users\Viknesh G\AppData\Local\Programs\Ollama\ollama.EXE` (Version `0.40.6`).
2. **Local Daemon Availability:** Active and responding to HTTP requests at `http://localhost:11434`.
3. **Hardware Acceleration:** Dedicated NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, Driver 592.82, CUDA 13.1) is present and fully leveraged by Ollama (100% offload to GPU via `llamacpp`).
4. **Physical Memory & Storage:** 23.64 GB physical RAM (10.85 GB available); 65.87 GB free disk on drive `D:\`; 28.71 GB free on drive `C:\`.
5. **Installed Models:** 3 local models discovered:
   - `llama3.2:latest` (3.2B parameters, 1.88 GB on disk, 2.6 GB VRAM footprint, Q4_K_M).
   - `llama3.2:1b` (1.2B parameters, 1.23 GB on disk, Q8_0).
   - `qwen2.5:1.5b` (1.5B parameters, 0.92 GB on disk, 1.2 GB VRAM footprint, Q4_K_M).
6. **Multimodal Capability Assessment:** None of the 3 installed models possess native vision encoders (the vision variant of Llama 3.2 is `llama3.2-vision:11b`). In strict adherence to Task C operating rules, all records will be processed via their text/dialogue modalities, while visual interpretation is formally marked as `UNASSESSED`. No vision understanding will be fabricated.

---

## 2. Environment Diagnostics Matrix

| Diagnostic Property | Observed Value | Verification Method | Status |
|---|---|---|---|
| **Operating System** | Windows 11 AMD64 (WDDM Driver Model) | Windows API (`kernel32`) | OK |
| **Python Runtime** | Python 3.12.10 (AMD64) | `sys.version` | OK |
| **Ollama Executable** | `C:\Users\Viknesh G\AppData\Local\Programs\Ollama\ollama.EXE` | `shutil.which('ollama')` | OK |
| **Ollama Version** | `0.40.6` | `ollama --version` | OK |
| **Local API Endpoint** | `http://localhost:11434` | HTTP GET `/api/tags` | OK |
| **Physical Memory (RAM)** | Total: 23.64 GB \| Available: 10.85 GB | `GlobalMemoryStatusEx` | OK |
| **Disk Space (D:)** | Total: 150.00 GB \| Free: 65.87 GB | `shutil.disk_usage` | OK |
| **Dedicated GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) | `nvidia-smi` | OK |
| **GPU Offload Support** | 100% GPU Layer Offload Verified | `ollama ps` | OK |
| **Python HTTP Client** | `httpx` & `requests` available | `import httpx, requests` | OK |

---

## 3. Discovered Model Inventory & Suitability Analysis

Querying `http://localhost:11434/api/tags` revealed the following installed models:

```json
[
  {
    "name": "llama3.2:latest",
    "size": 2019393189,
    "parameter_size": "3.2B",
    "quantization": "Q4_K_M",
    "family": "llama"
  },
  {
    "name": "llama3.2:1b",
    "size": 1321074811,
    "parameter_size": "1.2B",
    "quantization": "Q8_0",
    "family": "llama"
  },
  {
    "name": "qwen2.5:1.5b",
    "size": 986047754,
    "parameter_size": "1.5B",
    "quantization": "Q4_K_M",
    "family": "qwen2"
  }
]
```

### Model Selection Rationale (Task C):
1. **Primary Annotation Model:** `llama3.2:latest` (3.2B parameters).
   - High instruction-following adherence and nuanced pragmatic grasp of indirect speech.
   - Fits 100% within the 6 GB VRAM of the RTX 4050 (~2.6 GB footprint).
   - Verified execution latency: ~1.8 seconds per comment in warm cache.
2. **Secondary Pass Model:**
   - Supported options:
     - Independent repeat pass using `llama3.2:latest` with non-zero temperature or fresh prompt seed (repeated-run consistency).
     - Cross-model pass using `qwen2.5:1.5b` (cross-model agreement).

---

## 4. Modality & Vision Policy

> [!IMPORTANT]
> Because no vision-language model (e.g. `llama3.2-vision`, `llava`) is currently installed in Ollama:
> 1. All social media comments and dialogue context will be evaluated purely from text signals.
> 2. The pipeline will record `modality_used = "TEXT_ONLY"`.
> 3. For multimodal instances where an image exists, `image_role` will be recorded as `UNASSESSED_VISION_MODEL_ABSENT`.
> 4. The model will **never claim to have inspected or decoded image pixels**.
> 5. This maintains 100% scientific honesty without downloading unauthorized multi-gigabyte vision models or calling paid external vision APIs.
