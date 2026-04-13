from bridge.emotional_prompt import detect_emotional_state, build_emotional_prompt


def test_curious_state():
    mods = {"DA": 0.05, "NE": 0.02, "ACh": 0.03, "5HT": 0.02}
    name, state = detect_emotional_state(mods)
    assert name == "curious"


def test_alert_state():
    mods = {"DA": 0.02, "NE": 0.10, "ACh": 0.03, "5HT": 0.01}
    name, state = detect_emotional_state(mods)
    assert name == "alert"


def test_stressed_highest_priority():
    mods = {"DA": 0.05, "NE": 0.10, "ACh": 0.05, "5HT": 0.01}
    name, state = detect_emotional_state(mods)
    assert name == "stressed"


def test_drowsy_all_low():
    mods = {"DA": 0.001, "NE": 0.001, "ACh": 0.001, "5HT": 0.001}
    name, state = detect_emotional_state(mods)
    assert name == "drowsy"


def test_fallback_content():
    mods = {"DA": 0.01, "NE": 0.01, "ACh": 0.01, "5HT": 0.01}
    name, state = detect_emotional_state(mods)
    assert name == "content"


def test_prompt_contains_emotional_state():
    mods = {"DA": 0.05, "NE": 0.02, "ACh": 0.03, "5HT": 0.02}
    prompt = build_emotional_prompt(mods, "Tastatur: still", "Kein Muster")
    assert "CURIOUS" in prompt
    assert "NEUGIERIG" in prompt


def test_prompt_always_has_rules():
    mods = {"DA": 0.001, "NE": 0.001, "ACh": 0.001, "5HT": 0.001}
    prompt = build_emotional_prompt(mods, "test", "test")
    assert "KEINE Emojis" in prompt
    assert "ABSOLUTE REGELN" in prompt
