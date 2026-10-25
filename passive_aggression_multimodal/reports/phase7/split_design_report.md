# Phase 7 Reproducible Split Design Report

**Date:** October 21, 2026  
**Artifact:** `reports/phase7/split_design_report.md`  

## 1. Split Strategy Specification
- **Algorithm:** Group-Aware Deterministic MD5 Partitioning (Seed: `42`)
- **Total Input Records:** 27,970 (MMSD: 24,635 + SarcNet: 3,335)
- **Target Proportions:** 70% Train, 15% Validation, 15% Test
- **Actual Counts:**
  - **Train:** 19,618 (70.14%)
  - **Validation:** 4,143 (14.81%)
  - **Test:** 4,209 (15.05%)
- **Cross-Partition Duplicate Leakage:** **0 records**
