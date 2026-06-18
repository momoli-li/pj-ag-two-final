"""Deterministic mock LLM."""
import random
from typing import List, Optional


class DeterministicLLM:
    def __init__(self, seed: int = 42, extract_noise: float = 0.05,
                 no_context_correct: float = 0.05):
        self.seed = seed
        self.extract_noise = extract_noise
        self.no_context_correct = no_context_correct
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None):
        self._rng = random.Random(self.seed if seed is None else seed)

    def generate(self, question: str, short_term_context: str = "",
                 long_term_context: str = "",
                 ground_truth_keywords: Optional[List[str]] = None) -> str:
        ctx = (short_term_context + "\n" + long_term_context).strip()
        if ground_truth_keywords:
            seen = [kw for kw in ground_truth_keywords if kw and kw in ctx]
            if seen:
                if self._rng.random() < self.extract_noise:
                    return self._fallback()
                return self._answer_with(seen[0])
            else:
                if self._rng.random() < self.no_context_correct:
                    return self._answer_with(ground_truth_keywords[0])
                return self._fallback()
        return "好的，已了解。"

    def _answer_with(self, key: str) -> str:
        templates = [
            f"根据您之前提到的，应该是 {key}。",
            f"您前面说过相关信息：{key}。",
            f"基于上下文，我的回答是：{key}。",
            f"我记得是 {key}，对吗？",
            f"前文提到过 —— {key}。",
        ]
        return templates[self._rng.randrange(len(templates))]

    def _fallback(self) -> str:
        templates = [
            "抱歉，我没有这方面的记录。",
            "我目前没有相关上下文。",
            "这一信息我没有印象。",
            "我无法从已有对话中找到相关内容。",
        ]
        return templates[self._rng.randrange(len(templates))]
