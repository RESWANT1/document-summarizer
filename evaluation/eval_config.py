"""
evaluation/eval_config.py
Configuration for evaluation pipeline.
"""

EVAL_CONFIG = {
    "arxiv": {
        "enabled": True,
        "num_samples": 5,
        "split": "validation",
    },
    "multiformat": {
        "enabled": True,
        "formats": ["pdf", "docx", "image"],
    },
}

MODELS_TO_EVALUATE = ["textrank", "bart", "hla_mmr"]

METRICS_CONFIG = {
    "rouge": {
        "enabled": True,
        "types": ["rouge1", "rouge2", "rougeL"],
        "use_stemmer": True,
    },
    "summac": {
        "enabled": True,
        "granularity": "sentence",
        "model_name": "vitc",
    },
}

OUTPUT_CONFIG = {
    "save_model_outputs": True,
    "save_csv_results": True,
    "save_latex_tables": True,
    "save_charts": True,
    "save_analysis_report": True,
    "verbose_logging": True,
}

SUMMARIZATION_CONFIG = {
    "length_mode": "medium",
    "target_word_count": 0,
    "textrank": {
        "compression_ratio": 0.25,
    },
    "bart": {
        "num_beams": 4,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
    },
    "hla_mmr": {
        "extraction_ratio": 0.25,
        "mmr_lambda": 0.85,
        "centroid_weight": 0.6,
        "position_weight": 0.2,
    },
}

PERFORMANCE_CONFIG = {
    "use_gpu": True,
    "batch_size": 1,
    "max_input_tokens": 1024,
    "cache_embeddings": True,
    "num_workers": 0,
}
