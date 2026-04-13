"""Feedback Channel — LLM actions affect the SNN.

When the user confirms a label -> DA injection (reward signal)
When the user corrects -> NE injection (error signal)
When the user is engaged (fast replies) -> ACh boost
When the user goes silent -> 5HT decay

This closes the loop: SNN -> LLM -> User -> SNN
"""
from __future__ import annotations
import time
from typing import Any

from brain.core import Brain


class FeedbackChannel:
    """Translates user interactions into SNN modulator injections."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self._last_user_message_time: float = 0
        self._message_count: int = 0

    def on_user_message(self, message: str) -> None:
        """Called when the user sends a chat message."""
        now = time.time()
        gap = now - self._last_user_message_time if self._last_user_message_time > 0 else 999

        # Fast replies (< 30s) = engaged -> ACh boost
        if gap < 30:
            self.brain.modulators.inject("ACh", 0.005)
            self.brain.modulators.inject("DA", 0.002)

        # Very fast (< 10s) = excited conversation -> more DA
        if gap < 10:
            self.brain.modulators.inject("DA", 0.005)

        self._last_user_message_time = now
        self._message_count += 1

    def on_label_confirmed(self) -> None:
        """User confirmed a label suggestion -> reward."""
        self.brain.modulators.inject("DA", 0.02)   # strong reward
        self.brain.modulators.inject("5HT", 0.01)  # satisfaction

    def on_label_corrected(self) -> None:
        """User corrected a label -> error signal."""
        self.brain.modulators.inject("NE", 0.01)   # mild surprise
        self.brain.modulators.inject("DA", 0.005)   # still learning

    def on_negative_feedback(self) -> None:
        """User expressed dissatisfaction -> reduce 5HT."""
        self.brain.modulators.inject("NE", 0.015)
        self.brain.modulators.inject("5HT", -0.005)

    def on_positive_feedback(self) -> None:
        """User expressed satisfaction -> boost 5HT."""
        self.brain.modulators.inject("5HT", 0.015)
        self.brain.modulators.inject("DA", 0.01)
