"""Short-term memory: fixed-window FIFO."""
from collections import deque
from typing import List, Dict

class ShortTermMemory:
    def __init__(self, window_n: int = 10):
        self.n = window_n
        self.buffer = deque(maxlen=window_n)
    def append(self, turn: Dict[str, str]):
        self.buffer.append(turn)
    def get_context(self) -> List[Dict[str, str]]:
        return list(self.buffer)
    def render(self) -> str:
        if not self.buffer: return ""
        return "\n".join(f"{'用户' if t['role']=='user' else '助手'}: {t['text']}" for t in self.buffer)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)
