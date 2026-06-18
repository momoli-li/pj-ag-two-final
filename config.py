"""Hyperparameters for PJ-AG2. All knobs in one place."""

DEFAULT_CONFIG = {
    "short_window_n": 10,
    "embedding_type": "tfidf_char",
    "tfidf_ngram_range": (2, 4),
    "tfidf_max_features": 4096,
    "top_k": 5,
    "retrieval_threshold": 0.10,
    "llm_seed": 42,
    "llm_extract_noise": 0.05,
    "llm_no_context_correct": 0.05,
    "llm_pollution_amplify": 0.85,
    "summary_trigger_turns": 8,
}
