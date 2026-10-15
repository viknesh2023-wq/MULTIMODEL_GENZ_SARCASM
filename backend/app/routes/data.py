"""
Dataset and feature status routes.

GET /api/dataset/stats   — Text + multimodal dataset info
GET /api/features/status — Feature extraction pipeline status
"""
from fastapi import APIRouter
from app.schemas.schemas import DatasetStatsResponse, FeatureStatusResponse

router = APIRouter(prefix="/api", tags=["Data"])


@router.get("/dataset/stats", response_model=DatasetStatsResponse)
async def dataset_stats():
    return DatasetStatsResponse(
        text_dataset={
            "total_samples": 1005167,
            "train_samples": 812130,
            "validation_samples": 96518,
            "test_samples": 96519,
            "non_sarcastic": 499348,
            "sarcastic": 505819,
            "null_count": 0,
            "duplicate_count": 0,
            "languages": {
                "English": 985168,
                "Tamil-English": 6680,
                "Kannada-English": 6660,
                "Malayalam-English": 6659,
            },
            "sources": {
                "SARC (Reddit)": 956097,
                "GenZ-SarC": 19984,
                "Multilingual-Synthetic": 19999,
                "SarcasmCorpusV2": 9087,
            },
            "file_path": "FINAL_DATASETS\\TEXT_SARCASM\\text_sarcasm_combined.csv",
        },
        multimodal_dataset={
            "total_samples": 33077,
            "train_samples": 25384,
            "validation_samples": 3891,
            "test_samples": 3802,
            "non_sarcastic": 14568,
            "sarcastic": 18509,
            "null_count": 0,
            "duplicate_count": 0,
            "verified_images_on_disk": 33077,
            "clip_features_extracted": 33077,
            "internvl_features_extracted": 33077,
            "xlmr_features_extracted": 33077,
            "sentiment_features_extracted": 33077,
            "emotion_features_extracted": 33077,
            "sources": {
                "MMSD 2.0": 24635,
                "MEMOTION 3.0": 8442,
            },
            "file_path": "FINAL_DATASETS\\MULTIMODAL_SARCASM\\multimodal_sarcasm_combined.csv",
        },
        sentiment_dataset={
            "total_samples": 132324,
            "train_samples": 107678,
            "validation_samples": 7916,
            "test_samples": 16730,
            "positive": 61810,
            "neutral": 45405,
            "negative": 25109,
            "classes": 3,
            "null_count": 0,
            "duplicate_count": 0,
            "sources": {
                "TweetEval": 59305,
                "SentiMix": 38019,
                "DravidianCodeMix": 23000,
                "Synthetic": 12000,
            },
            "file_path": "FINAL_DATASETS\\SENTIMENT\\sentiment_combined.csv",
        },
        emotion_dataset={
            "total_samples": 54263,
            "train_samples": 43410,
            "validation_samples": 5426,
            "test_samples": 5427,
            "total_classes": 28,
            "taxonomy": "Google GoEmotions",
            "multi_label_samples": 8817,
            "null_count": 0,
            "sources": {
                "GoEmotions (Reddit)": 54263,
            },
            "file_path": "FINAL_DATASETS\\EMOTION\\emotion_combined.csv",
        },
    )


@router.get("/features/status", response_model=FeatureStatusResponse)
async def features_status():
    return FeatureStatusResponse(
        features=[
            {
                "name": "XLM-R Text",
                "dimension": 768,
                "samples": 33077,
                "status": "ready",
                "status_label": "FEATURES READY",
                "color": "green",
                "description": "768-dimensional contextual text representation from XLM-R encoder",
            },
            {
                "name": "CLIP Vision",
                "dimension": 512,
                "samples": 33077,
                "status": "ready",
                "status_label": "FEATURES READY",
                "color": "green",
                "description": "512-dimensional image embedding from OpenAI CLIP ViT-B/32",
            },
            {
                "name": "InternVL3-2B",
                "dimension": 1536,
                "samples": 33077,
                "status": "ready",
                "status_label": "FEATURES READY",
                "color": "green",
                "description": "1536-dimensional vision-language representation from InternVL3-2B",
            },
            {
                "name": "Sentiment",
                "dimension": 3,
                "samples": 33077,
                "status": "ready",
                "status_label": "FEATURES READY",
                "color": "green",
                "description": "3-dimensional sentiment probability vector (positive/neutral/negative)",
            },
            {
                "name": "Emotion",
                "dimension": 28,
                "samples": 33077,
                "status": "ready",
                "status_label": "FEATURES READY",
                "color": "green",
                "description": "28-dimensional emotion probability vector (GoEmotions taxonomy)",
            },
            {
                "name": "Fusion Classifier",
                "dimension": 2847,
                "samples": 33077,
                "status": "ready",
                "status_label": "MODEL READY",
                "color": "green",
                "description": (
                    "Final 2847-D fusion: XLM-R(768) + CLIP(512) + InternVL(1536) "
                    "+ Sentiment(3) + Emotion(28) = 2847-D"
                ),
            },
        ]
    )
