"""Agent: orchestrates short + long memory and the LLM."""
from typing import List, Dict, Optional
from short_term_memory import ShortTermMemory
from long_term_memory import LongTermMemory
from llm import DeterministicLLM


class Agent:
    MODES = ("no_memory", "short_only", "short_long")

    def __init__(self, mode: str, llm, short_term=None, long_term=None):
        assert mode in self.MODES
        self.mode = mode
        self.llm = llm
        self.short_term = short_term
        self.long_term = long_term
        self.turn_index = 0

    def observe_user(self, text: str):
        self.turn_index += 1
        if self.mode in ("short_only", "short_long") and self.short_term is not None:
            self.short_term.append({"role": "user", "text": text})
        if self.mode == "short_long" and self.long_term is not None:
            self.long_term.write(text, meta={"turn": self.turn_index, "role": "user"})

    def observe_agent(self, text: str):
        if self.mode in ("short_only", "short_long") and self.short_term is not None:
            self.short_term.append({"role": "agent", "text": text})

    def respond(self, question: str, ground_truth_keywords=None):
        short_ctx = ""
        long_ctx = ""
        if self.mode == "short_only":
            short_ctx = self.short_term.render() if self.short_term else ""
        elif self.mode == "short_long":
            short_ctx = self.short_term.render() if self.short_term else ""
            long_ctx = self.long_term.render(question) if self.long_term else ""
        resp = self.llm.generate(
            question=question, short_term_context=short_ctx,
            long_term_context=long_ctx, ground_truth_keywords=ground_truth_keywords,
        )
        return {"response": resp, "short_ctx": short_ctx, "long_ctx": long_ctx}
