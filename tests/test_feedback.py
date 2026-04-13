"""Tests for bridge.feedback — FeedbackChannel.

Uses MagicMock for Brain so tests run without torch or a real SNN.
Every test verifies modulators.inject() is called with the correct
modulator name and amount.
"""
from __future__ import annotations
import time
from unittest.mock import MagicMock, call

import pytest

from bridge.feedback import FeedbackChannel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_brain():
    """Create a mock Brain with a mock modulators attribute."""
    brain = MagicMock()
    brain.modulators = MagicMock()
    return brain


@pytest.fixture
def channel(mock_brain):
    """Create a FeedbackChannel backed by the mock brain."""
    return FeedbackChannel(mock_brain)


# ---------------------------------------------------------------------------
# on_label_confirmed
# ---------------------------------------------------------------------------

class TestLabelConfirmed:
    def test_injects_da_reward(self, channel, mock_brain):
        channel.on_label_confirmed()
        mock_brain.modulators.inject.assert_any_call("DA", 0.02)

    def test_injects_5ht_satisfaction(self, channel, mock_brain):
        channel.on_label_confirmed()
        mock_brain.modulators.inject.assert_any_call("5HT", 0.01)

    def test_exactly_two_injections(self, channel, mock_brain):
        channel.on_label_confirmed()
        assert mock_brain.modulators.inject.call_count == 2


# ---------------------------------------------------------------------------
# on_label_corrected
# ---------------------------------------------------------------------------

class TestLabelCorrected:
    def test_injects_ne_surprise(self, channel, mock_brain):
        channel.on_label_corrected()
        mock_brain.modulators.inject.assert_any_call("NE", 0.01)

    def test_injects_da_learning(self, channel, mock_brain):
        channel.on_label_corrected()
        mock_brain.modulators.inject.assert_any_call("DA", 0.005)

    def test_exactly_two_injections(self, channel, mock_brain):
        channel.on_label_corrected()
        assert mock_brain.modulators.inject.call_count == 2


# ---------------------------------------------------------------------------
# on_negative_feedback
# ---------------------------------------------------------------------------

class TestNegativeFeedback:
    def test_injects_ne_alert(self, channel, mock_brain):
        channel.on_negative_feedback()
        mock_brain.modulators.inject.assert_any_call("NE", 0.015)

    def test_injects_negative_5ht(self, channel, mock_brain):
        channel.on_negative_feedback()
        mock_brain.modulators.inject.assert_any_call("5HT", -0.005)

    def test_exactly_two_injections(self, channel, mock_brain):
        channel.on_negative_feedback()
        assert mock_brain.modulators.inject.call_count == 2


# ---------------------------------------------------------------------------
# on_positive_feedback
# ---------------------------------------------------------------------------

class TestPositiveFeedback:
    def test_injects_5ht_boost(self, channel, mock_brain):
        channel.on_positive_feedback()
        mock_brain.modulators.inject.assert_any_call("5HT", 0.015)

    def test_injects_da_reward(self, channel, mock_brain):
        channel.on_positive_feedback()
        mock_brain.modulators.inject.assert_any_call("DA", 0.01)

    def test_exactly_two_injections(self, channel, mock_brain):
        channel.on_positive_feedback()
        assert mock_brain.modulators.inject.call_count == 2


# ---------------------------------------------------------------------------
# on_user_message — slow reply (gap >= 30s), no modulator injection
# ---------------------------------------------------------------------------

class TestUserMessageSlow:
    def test_first_message_no_injection(self, channel, mock_brain):
        """First message ever: gap is infinite, no engagement signal."""
        channel.on_user_message("hello")
        mock_brain.modulators.inject.assert_not_called()

    def test_slow_reply_no_injection(self, channel, mock_brain):
        """Reply after 60s: not engaged, no injection."""
        channel._last_user_message_time = time.time() - 60
        channel.on_user_message("still here")
        mock_brain.modulators.inject.assert_not_called()

    def test_increments_message_count(self, channel, mock_brain):
        channel.on_user_message("one")
        channel.on_user_message("two")
        assert channel._message_count == 2


# ---------------------------------------------------------------------------
# on_user_message — fast reply (gap < 30s) -> ACh + DA
# ---------------------------------------------------------------------------

class TestUserMessageFast:
    def test_fast_reply_injects_ach(self, channel, mock_brain):
        """Reply within 20s triggers ACh engagement boost."""
        channel._last_user_message_time = time.time() - 20
        channel.on_user_message("quick reply")
        mock_brain.modulators.inject.assert_any_call("ACh", 0.005)

    def test_fast_reply_injects_da(self, channel, mock_brain):
        """Reply within 20s also triggers small DA boost."""
        channel._last_user_message_time = time.time() - 20
        channel.on_user_message("quick reply")
        mock_brain.modulators.inject.assert_any_call("DA", 0.002)

    def test_fast_reply_exactly_two_injections(self, channel, mock_brain):
        """Gap 20s: ACh + DA, but not the extra DA for very-fast."""
        channel._last_user_message_time = time.time() - 20
        channel.on_user_message("quick reply")
        assert mock_brain.modulators.inject.call_count == 2


# ---------------------------------------------------------------------------
# on_user_message — very fast reply (gap < 10s) -> ACh + DA + extra DA
# ---------------------------------------------------------------------------

class TestUserMessageVeryFast:
    def test_very_fast_injects_extra_da(self, channel, mock_brain):
        """Reply within 5s triggers the extra DA=0.005 excitement boost."""
        channel._last_user_message_time = time.time() - 5
        channel.on_user_message("super fast!")
        mock_brain.modulators.inject.assert_any_call("DA", 0.005)

    def test_very_fast_also_injects_ach(self, channel, mock_brain):
        """Very fast is also fast, so ACh is injected too."""
        channel._last_user_message_time = time.time() - 5
        channel.on_user_message("super fast!")
        mock_brain.modulators.inject.assert_any_call("ACh", 0.005)

    def test_very_fast_three_injections(self, channel, mock_brain):
        """Gap <10s: ACh(0.005) + DA(0.002) + DA(0.005) = 3 calls."""
        channel._last_user_message_time = time.time() - 5
        channel.on_user_message("super fast!")
        assert mock_brain.modulators.inject.call_count == 3

    def test_very_fast_injection_order(self, channel, mock_brain):
        """Verify the exact call sequence for very fast replies."""
        channel._last_user_message_time = time.time() - 5
        channel.on_user_message("super fast!")
        expected = [
            call("ACh", 0.005),
            call("DA", 0.002),
            call("DA", 0.005),
        ]
        assert mock_brain.modulators.inject.call_args_list == expected


# ---------------------------------------------------------------------------
# on_user_message — timestamp tracking
# ---------------------------------------------------------------------------

class TestUserMessageTimestamp:
    def test_updates_last_message_time(self, channel, mock_brain):
        before = time.time()
        channel.on_user_message("check time")
        after = time.time()
        assert before <= channel._last_user_message_time <= after

    def test_second_fast_message_uses_updated_time(self, channel, mock_brain):
        """After two rapid messages, the second should use the first's time."""
        channel.on_user_message("first")
        t1 = channel._last_user_message_time

        # Immediately send another — gap ~0s, so very fast
        channel.on_user_message("second")
        t2 = channel._last_user_message_time

        assert t2 >= t1
        # Second message should have triggered injections (gap ~0s < 10s)
        assert mock_brain.modulators.inject.call_count == 3  # ACh + DA + DA
