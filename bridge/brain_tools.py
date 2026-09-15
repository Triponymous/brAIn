"""BrainTools — the brain as tools the LLM calls while it reasons.

README, The Thesis: the LLM understands the organism completely, as tools,
not as a paragraph. Nothing here is new state. Every tool is a read of what
already exists — the felt-state model, the modulators and the prediction-
error driver behind them, the concept tracker, the episode log, the habit
miner and anomaly detector, the synapse explainer, the experience log —
projected into answers a language model can reason over.

Tool names are identifiers (brain_state, not brain.state): Ollama and
OpenAI-style function calling require it. Definitions use the Anthropic
input_schema shape the rest of the registry uses; llm_local converts.
"""
from __future__ import annotations
import datetime as dt
import time
from collections import Counter
from typing import Any

from brain.core import Brain
from bridge.felt_state import SIGNATURE_KEYS

_MODS = ("DA", "NE", "ACh", "5HT")
_TICK_HZ = 100.0
_MECHANISM = {
    "NE": "surprise: prediction error above its own recent baseline (pe_fast - pe_slow)",
    "DA": "learning progress: prediction error dropping (pe_slow - pe_fast), gated by precision so pure noise does not count",
    "ACh": "attention on novelty: the same progress signal, injected only while the human is active",
    "5HT": "calm: low surprise and low variance sustained; earned continuously, baseline zero",
}


def _when(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _r(x: Any, nd: int = 4) -> Any:
    return round(float(x), nd) if isinstance(x, (int, float)) and not isinstance(x, bool) else x


def _pattern_name(row: dict[str, Any]) -> str:
    ss = row.get("sensor_summary") or {}
    label = ss.get("cluster_label")
    cid = ss.get("cluster_id", -1)
    return label or ("none" if cid is None or cid < 0 else f"#{cid}")


class BrainTools:
    NAMES = frozenset({
        "brain_state", "brain_history", "brain_concept", "brain_felt", "brain_habits",
        "brain_anomalies", "brain_why", "brain_recall", "brain_experience",
    })

    def __init__(self, brain: Brain, *, episodes: Any = None, experience: Any = None,
                 interpreter: Any = None, narrator: Any = None) -> None:
        self.brain = brain
        self.episodes = episodes          # EpisodeLogger: one snapshot per ~10 s
        self.experience = experience      # ExperienceLog: what happened between us
        self.interpreter = interpreter    # BrainInterpreter: habits, anomalies, explainer
        self.narrator = narrator          # SNNNarrator: the brain's own words

    # ------------------------------------------------------------------ tools

    def brain_state(self) -> dict[str, Any]:
        b = self.brain
        mods = b.modulators.snapshot()
        sig = getattr(b, "_last_signature", None)
        fs = getattr(b, "felt_state", None)
        cluster = getattr(b, "_last_concept_cluster", None)
        felt: dict[str, Any] = {"label": None, "confidence": 0.0, "known_labels": []}
        if fs is not None:
            felt["known_labels"] = fs.known_labels()
            if sig:
                felt["label"], felt["confidence"] = fs.recognize(sig, cluster)
        snap = b.concept_tracker.snapshot()
        cid = snap.get("current_cluster", -1)
        c = b.concept_tracker._clusters.get(cid) if cid is not None and cid >= 0 else None
        sd = getattr(b, "_last_sensor_display", None) or {}
        wm = b.regions.get("wm")
        wm_active = int(wm.last_spikes.sum().item()) if wm is not None and hasattr(wm, "last_spikes") else 0
        return {
            "felt": felt,
            "modulators": {k: _r(v) for k, v in mods.items()},
            "trend_180s": dict(zip(SIGNATURE_KEYS, (_r(x) for x in sig))) if sig else None,
            "prediction_error": self._prediction_error(),
            "pattern": {"cluster_id": cid, "label": snap.get("current_label"),
                        "times_seen": c.count if c else 0,
                        "confidence": _r((snap.get("debug") or {}).get("best_sim", 0.0), 2)},
            "senses": {"app": sd.get("app"), "keys_per_s": sd.get("keys"), "mouse_per_s": sd.get("mouse"),
                       "idle_s": sd.get("idle"), "mic_rms": sd.get("mic_rms"),
                       "app_switches_per_min": sd.get("switch_rate")},
            "working_memory_active": wm_active,
            "sleeping": bool(b.sleep_mode),
            "age": {"ticks": b.tick_count, "days": _r(b.tick_count / (_TICK_HZ * 86400), 2)},
        }

    def brain_history(self, hours: float = 24.0, step_minutes: float = 30.0) -> dict[str, Any]:
        if self.episodes is None:
            return {"error": "no episode log"}
        hours = max(0.1, float(hours))
        step = max(1.0, float(step_minutes)) * 60
        since = time.time() - hours * 3600
        rows = sorted(self.episodes.query(last_n=100_000, since_timestamp=since), key=lambda r: r["timestamp"])
        buckets: dict[int, list[dict]] = {}
        for r in rows:
            buckets.setdefault(int((r["timestamp"] - since) // step), []).append(r)
        out = []
        for i in sorted(buckets):
            rs = buckets[i]
            patterns = Counter(_pattern_name(r) for r in rs)
            apps = Counter(r["sensor_summary"].get("app") for r in rs if r["sensor_summary"].get("app"))
            out.append({
                "start": _when(since + i * step), "end": _when(since + (i + 1) * step), "samples": len(rs),
                "pattern": patterns.most_common(1)[0][0], "patterns": dict(patterns.most_common(3)),
                "apps": [a for a, _ in apps.most_common(3)],
                "modulators": {k: _r(sum(r["modulators"].get(k, 0.0) for r in rs) / len(rs)) for k in _MODS},
                "asleep": _r(sum(1 for r in rs if r["sleep_mode"]) / len(rs), 2),
            })
        return {"hours": hours, "step_minutes": step / 60, "samples_every_s": 10, "buckets": out}

    def brain_concept(self, cluster_id: int) -> dict[str, Any]:
        ct = self.brain.concept_tracker
        cid = int(cluster_id)
        c = ct._clusters.get(cid)
        if c is None:
            return {"error": f"unknown cluster {cid}",
                    "known": [{"cluster_id": k, "label": v.label} for k, v in sorted(ct._clusters.items())]}
        out: dict[str, Any] = {
            "cluster_id": cid, "label": c.label, "times_seen": c.count,
            "last_seen_s_ago": int(max(0, self.brain.tick_count - c.last_seen) / _TICK_HZ),
            "active_now": ct.current_cluster == cid,
        }
        if self.interpreter is not None:
            out["senses"] = self.interpreter.synapse_explainer.concept_profile(cid).get("active_sensors", [])
        if self.episodes is not None:
            rows = self.episodes.query(last_n=100_000, since_timestamp=time.time() - 7 * 86400)
            n = sum(1 for r in rows if (r.get("sensor_summary") or {}).get("cluster_id") == cid)
            out["minutes_last_7_days"] = _r(n * 10 / 60, 1)
        return out

    def brain_felt(self, label: str | None = None) -> dict[str, Any]:
        fs = getattr(self.brain, "felt_state", None)
        if fs is None:
            return {"error": "no felt-state model yet"}
        protos = fs.to_dict()["prototypes"]
        sig = getattr(self.brain, "_last_signature", None)
        now_label, conf = (fs.recognize(sig, getattr(self.brain, "_last_concept_cluster", None))
                           if sig else (None, 0.0))
        if not label:
            return {"now": {"label": now_label, "confidence": conf},
                    "known": [{"label": k, "times_taught": v["count"], "cluster": v.get("cluster")}
                              for k, v in sorted(protos.items())]}
        key = next((k for k in protos if k.lower() == str(label).lower()), None)
        if key is None:
            return {"error": f"unknown state '{label}'", "known": sorted(protos)}
        p = protos[key]
        out: dict[str, Any] = {
            "label": key, "times_taught": p["count"], "cluster": p.get("cluster"),
            "signature": dict(zip(SIGNATURE_KEYS, (_r(x) for x in p["centroid"]))),
            "recognized_now": now_label == key,
        }
        if self.experience is not None:
            taught = [r for r in self.experience.recent(limit=1000, since=time.time() - 30 * 86400)
                      if r["kind"] == "teach_felt" and str(r["payload"].get("label", "")).lower() == key.lower()]
            out["taught_at"] = [_when(r["ts"]) for r in taught[:10]]
        return out

    def brain_habits(self) -> dict[str, Any]:
        if self.interpreter is None:
            return {"error": "no habit miner"}
        hm = self.interpreter.habit_miner
        return {"now": hm.current_hour_context(), "habits": hm.current_habits()}

    def brain_anomalies(self) -> dict[str, Any]:
        if self.interpreter is None:
            return {"error": "no anomaly detector"}
        sd = getattr(self.brain, "_last_sensor_display", None) or {}
        return {"anomalies": self.interpreter.anomaly.check(sd)}

    def brain_why(self, modulator: str | None = None) -> dict[str, Any]:
        mods = self.brain.modulators.snapshot()
        readings = self.interpreter.synapse_explainer.explain_modulators() if self.interpreter is not None else {}
        out: dict[str, Any] = {"prediction_error": self._prediction_error()}
        if self.narrator is not None:
            out["narrative"] = self.narrator.narrate_full_state()
        if modulator:
            key = {"da": "DA", "ne": "NE", "ach": "ACh", "5ht": "5HT", "sht": "5HT"}.get(str(modulator).lower())
            if key is None:
                return {"error": f"unknown modulator '{modulator}'", "known": list(_MODS)}
            out.update({"modulator": key, "level": _r(mods.get(key, 0.0)), "reading": readings.get(key),
                        "mechanism": _MECHANISM[key]})
            return out
        out["modulators"] = {k: {"level": _r(mods.get(k, 0.0)), "reading": readings.get(k),
                                 "mechanism": _MECHANISM[k]} for k in _MODS}
        return out

    def brain_recall(self, hours: float = 24.0, label: str | None = None,
                     min_minutes: float = 5.0) -> dict[str, Any]:
        if self.episodes is None:
            return {"error": "no episode log"}
        hours = max(0.1, float(hours))
        rows = sorted(self.episodes.query(last_n=100_000, since_timestamp=time.time() - hours * 3600),
                      key=lambda r: r["timestamp"])
        spans: list[dict[str, Any]] = []
        for r in rows:
            name = _pattern_name(r)
            app = (r.get("sensor_summary") or {}).get("app")
            last = spans[-1] if spans else None
            # samples land every ~10 s; a gap under 90 s is the same stretch
            if last is not None and last["pattern"] == name and r["timestamp"] - last["end"] <= 90:
                last["end"] = r["timestamp"]
                if app:
                    last["apps"][app] += 1
            else:
                spans.append({"pattern": name, "start": r["timestamp"], "end": r["timestamp"],
                              "apps": Counter([app] if app else [])})
        out = []
        for s in spans:
            minutes = (s["end"] - s["start"]) / 60 + 10 / 60   # a lone sample still stands for ~10 s
            if minutes < float(min_minutes):
                continue
            if label and str(label).lower() not in s["pattern"].lower():
                continue
            out.append({"pattern": s["pattern"], "start": _when(s["start"]), "end": _when(s["end"]),
                        "minutes": _r(minutes, 1), "apps": [a for a, _ in s["apps"].most_common(3)]})
        longest = max(out, key=lambda s: s["minutes"]) if out else None
        return {"hours": hours, "label": label, "spans": out[-20:], "longest": longest}

    def brain_experience(self, hours: float = 24.0) -> dict[str, Any]:
        if self.experience is None:
            return {"error": "no experience log"}
        hours = max(0.1, float(hours))
        recent = [{"when": _when(r["ts"]), "actor": r["actor"], "kind": r["kind"], "felt": r["felt_label"],
                   "response": r["response_kind"], "payload": r["payload"]}
                  for r in self.experience.recent(limit=10, since=time.time() - hours * 3600)]
        return {"summary": self.experience.summary(since_hours=hours), "recent": recent}

    # -------------------------------------------------------------- plumbing

    def _prediction_error(self) -> dict[str, Any] | None:
        b = self.brain
        fast, slow, var = (getattr(b, k, None) for k in ("_pe_fast", "_pe_slow", "_pe_var"))
        if fast is None or slow is None:
            return None
        return {"fast": _r(fast, 5), "slow": _r(slow, 5), "variance": _r(var or 0.0, 7),
                "surprise": _r(max(0.0, fast - slow), 5), "progress": _r(max(0.0, slow - fast), 5)}

    def execute(self, name: str, args: dict[str, Any] | None = None) -> Any:
        """Dispatch a tool call by name. Arguments come from a language model,
        so a bad set of them is an answer, not a crash."""
        if name not in self.NAMES:
            return {"error": f"Unknown brain tool: {name}"}
        try:
            return getattr(self, name)(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}

    @staticmethod
    def tool_definitions() -> list[dict[str, Any]]:
        no_args = {"type": "object", "properties": {}}
        hours = {"type": "number", "description": "How many hours back to look.", "default": 24}
        return [
            {"name": "brain_state",
             "description": "How I am right now: my felt-state and how sure I am, my chemistry (DA NE ACh 5HT) "
                            "with its 180-second trend and the prediction error behind it, the behaviour pattern "
                            "I recognise, what my senses show, whether I am asleep, my age. Call this before "
                            "answering anything about my current state.",
             "input_schema": no_args},
            {"name": "brain_history",
             "description": "What happened over the last N hours, in steps: the dominant pattern, apps, average "
                            "chemistry, sleep. For 'how was today', 'what did I do this morning'.",
             "input_schema": {"type": "object", "properties": {
                 "hours": hours,
                 "step_minutes": {"type": "number", "description": "Bucket size in minutes.", "default": 30}}}},
            {"name": "brain_concept",
             "description": "Everything about one behaviour pattern (cluster): label, how often seen, when last, "
                            "which senses define it, minutes in the last 7 days.",
             "input_schema": {"type": "object", "properties": {"cluster_id": {"type": "integer"}},
                              "required": ["cluster_id"]}},
            {"name": "brain_felt",
             "description": "My learned states, the words the human taught me. Without label: all of them and "
                            "which one I recognise now. With label: that state's signature, how often and when "
                            "it was taught, whether I am in it now.",
             "input_schema": {"type": "object", "properties": {"label": {"type": "string"}}}},
            {"name": "brain_habits",
             "description": "What usually happens at this hour and weekday, and my learned habits (app by hour, "
                            "confidence).",
             "input_schema": no_args},
            {"name": "brain_anomalies",
             "description": "What is unusual right now compared to the human's habits.",
             "input_schema": no_args},
            {"name": "brain_why",
             "description": "Why I feel the way I do: each modulator's level and reading, the prediction-error "
                            "mechanism behind it, and my own narrative. Optionally one modulator: DA, NE, ACh, 5HT.",
             "input_schema": {"type": "object", "properties": {"modulator": {"type": "string"}}}},
            {"name": "brain_recall",
             "description": "Find stretches of time in the last N hours when a pattern held: start, end, minutes, "
                            "apps. Optionally only patterns whose name contains label. For 'when was I last in "
                            "flow', 'how long did I code today'.",
             "input_schema": {"type": "object", "properties": {
                 "hours": hours,
                 "label": {"type": "string", "description": "Pattern name to look for, e.g. coding."},
                 "min_minutes": {"type": "number", "description": "Ignore stretches shorter than this.", "default": 5}}}},
            {"name": "brain_experience",
             "description": "What I did (asks, notifications), how the human responded, and how my chemistry "
                            "changed afterwards: the record of our interactions over the last N hours.",
             "input_schema": {"type": "object", "properties": {"hours": hours}}},
        ]
