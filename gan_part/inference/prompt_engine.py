"""
prompt_engine.py
----------------
Generates and manages text prompts for room interior synthesis.
"""

from __future__ import annotations

from ..data.room_class_map import ROOM_BASE_PROMPTS, NEGATIVE_PROMPT

class PromptEngine:
    """
    Engine for building high-quality text prompts tailored for specific room types.
    """

    def __init__(self, use_negative: bool = True):
        self.use_negative = use_negative

    def build(self, room_label: str, user_prompt: str | None = None) -> str:
        """
        Build a final prompt string.
        """
        base = ROOM_BASE_PROMPTS.get(room_label, ROOM_BASE_PROMPTS["default"])
        if user_prompt:
            return f"{user_prompt}, {base}"
        return base

    def get_negative(self) -> str | None:
        """Return the standard negative prompt."""
        return NEGATIVE_PROMPT if self.use_negative else None
