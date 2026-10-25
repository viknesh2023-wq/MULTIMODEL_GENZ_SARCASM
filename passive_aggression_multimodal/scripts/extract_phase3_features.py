"""Phase 3 Multimodal Feature Extraction Script.

Extracts 768-dimensional multimodal representations (comment, image, context, OCR)
using real pretrained text encoders (FacebookAI/xlm-roberta-base) and vision encoders,
caching verified feature batches into data/features/phase3/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_store import FeatureStore
from src.models.classification_head import PRIMARY_LABEL_TO_IDX, SARCASM_LABEL_TO_IDX
from src.models.text_encoder import PretrainedTextEncoder
from src.models.vision_encoder import PretrainedVisionEncoder


def extract_dataset_features(
    input_file: str,
    output_name: str,
    max_records: Optional[int] = None,
    batch_size: int = 16,
    label_type: str = "primary",  # "primary" or "sarcasm"
) -> Dict[str, Any]:
    """Extracts and stores multimodal features for a given jsonl manifest."""
    start_time = time.time()
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[*] Initializing feature extractors on CPU...")
    text_encoder = PretrainedTextEncoder(
        model_name_or_path="FacebookAI/xlm-roberta-base",
        max_length=128,
        pooling="mean",
        device="cpu",
        local_files_only=True,
    )
    vision_encoder = PretrainedVisionEncoder(
        model_name="openai/clip-vit-base-patch32",
        output_dim=768,
        device="cpu",
        local_files_only=True,
    )

    feature_store = FeatureStore(storage_dir="data/features/phase3")

    # Read records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records and len(records) >= max_records:
                break

    total_records = len(records)
    print(f"[*] Processing {total_records} records in batches of {batch_size}...")

    all_ids: List[str] = []
    all_comments: List[torch.Tensor] = []
    all_images: List[torch.Tensor] = []
    all_contexts: List[torch.Tensor] = []
    all_ocrs: List[torch.Tensor] = []
    all_img_masks: List[torch.Tensor] = []
    all_ctx_masks: List[torch.Tensor] = []
    all_ocr_masks: List[torch.Tensor] = []
    all_labels: List[int] = []

    for i in range(0, total_records, batch_size):
        batch = records[i : i + batch_size]
        b_ids = [r.get("example_id", f"rec_{i+j}") for j, r in enumerate(batch)]
        all_ids.extend(b_ids)

        # 1. Comment texts
        b_comments = [r.get("target_comment") or "<empty>" for r in batch]
        comment_emb = text_encoder(b_comments)
        all_comments.append(comment_emb)

        # 2. Images
        b_img_paths = [r.get("image_path") for r in batch]
        img_emb, img_mask = vision_encoder(b_img_paths)
        all_images.append(img_emb)
        all_img_masks.append(img_mask)

        # 3. Context
        b_contexts = []
        b_ctx_mask = []
        for r in batch:
            ctx = r.get("conversation_context") or r.get("caption") or ""
            if isinstance(ctx, list):
                ctx = " ".join(str(x) for x in ctx)
            has_ctx = bool(ctx and ctx.strip())
            b_contexts.append(ctx if has_ctx else "<no_context>")
            b_ctx_mask.append(has_ctx)

        ctx_emb = text_encoder(b_contexts)
        ctx_mask_tensor = torch.tensor(b_ctx_mask, dtype=torch.bool)
        # zero out if missing
        ctx_emb = ctx_emb * ctx_mask_tensor.unsqueeze(-1).float()
        all_contexts.append(ctx_emb)
        all_ctx_masks.append(ctx_mask_tensor)

        # 4. OCR
        b_ocrs = []
        b_ocr_mask = []
        for r in batch:
            ocr = r.get("image_ocr_text") or ""
            has_ocr = bool(ocr and ocr.strip())
            b_ocrs.append(ocr if has_ocr else "<no_ocr>")
            b_ocr_mask.append(has_ocr)

        ocr_emb = text_encoder(b_ocrs)
        ocr_mask_tensor = torch.tensor(b_ocr_mask, dtype=torch.bool)
        ocr_emb = ocr_emb * ocr_mask_tensor.unsqueeze(-1).float()
        all_ocrs.append(ocr_emb)
        all_ocr_masks.append(ocr_mask_tensor)

        # 5. Labels
        for r in batch:
            if label_type == "primary":
                lbl_str = r.get("passive_aggression_label")
                lbl_idx = PRIMARY_LABEL_TO_IDX.get(lbl_str, -1)
            else:  # sarcasm
                lbl_str = r.get("original_label")
                if lbl_str == "1":
                    lbl_idx = SARCASM_LABEL_TO_IDX["sarcastic"]
                elif lbl_str == "0":
                    lbl_idx = SARCASM_LABEL_TO_IDX["non_sarcastic"]
                else:
                    lbl_idx = -1
            all_labels.append(lbl_idx)

    # Cat batches
    comment_tensor = torch.cat(all_comments, dim=0)
    image_tensor = torch.cat(all_images, dim=0)
    context_tensor = torch.cat(all_contexts, dim=0)
    ocr_tensor = torch.cat(all_ocrs, dim=0)
    img_mask_tensor = torch.cat(all_img_masks, dim=0)
    ctx_mask_tensor = torch.cat(all_ctx_masks, dim=0)
    ocr_mask_tensor = torch.cat(all_ocr_masks, dim=0)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)

    # Save to feature store
    elapsed = time.time() - start_time
    saved_path = feature_store.save_features(
        name=output_name,
        example_ids=all_ids,
        comment_features=comment_tensor,
        image_features=image_tensor,
        context_features=context_tensor,
        ocr_features=ocr_tensor,
        image_masks=img_mask_tensor,
        context_masks=ctx_mask_tensor,
        ocr_masks=ocr_mask_tensor,
        labels=label_tensor,
        metadata={
            "input_file": str(input_path),
            "label_type": label_type,
            "total_records": total_records,
            "text_encoder": text_encoder.model_name_or_path,
            "vision_dim": image_tensor.shape[-1],
            "text_dim": comment_tensor.shape[-1],
            "extraction_time_sec": elapsed,
        },
    )

    result = {
        "output_name": output_name,
        "saved_path": str(saved_path),
        "total_records": total_records,
        "comment_shape": list(comment_tensor.shape),
        "image_shape": list(image_tensor.shape),
        "image_available_ratio": float(img_mask_tensor.float().mean().item()),
        "context_available_ratio": float(ctx_mask_tensor.float().mean().item()),
        "ocr_available_ratio": float(ocr_mask_tensor.float().mean().item()),
        "elapsed_seconds": elapsed,
        "records_per_second": total_records / max(elapsed, 0.001),
    }
    print(f"[+] Extraction complete in {elapsed:.2f}s ({result['records_per_second']:.1f} rec/s). Saved to {saved_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Phase 3 Multimodal Features")
    parser.add_argument("--input", type=str, default="data/synthetic/synthetic_pilot_350.jsonl")
    parser.add_argument("--name", type=str, default="synthetic_pilot_350")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--label_type", type=str, default="primary")
    args = parser.parse_args()

    extract_dataset_features(
        input_file=args.input,
        output_name=args.name,
        max_records=args.max_records,
        batch_size=args.batch_size,
        label_type=args.label_type,
    )
