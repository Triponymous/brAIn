"""Offline feature probe: python -m benchmark.audio_vad --help.

Compares the SAME fixed supervised probe with Mel/RMS vs Mel/RMS + VAD.
It does not train/evaluate the SNN, infer flow, or activate any sensor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from benchmark.audio_corpus import LABELS, SAMPLE_RATE, load_corpus, read_wav
from benchmark.audio_features import LocalVad, mel_features, vad_features


def predict(train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    mean, scale = train.mean(axis=0), train.std(axis=0)
    # A train-constant feature is unknown, not an infinitely strong test-time signal.
    variable = scale > 1e-8
    train = (train[:, variable] - mean[variable]) / scale[variable]
    test = (test[:, variable] - mean[variable]) / scale[variable]
    centers = np.stack([train[labels == index].mean(axis=0) for index in range(len(LABELS))])
    distances = ((test[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def metrics(truth: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(matrix, (truth, predictions), 1)
    tp = matrix.diagonal()
    support = matrix.sum(axis=1)
    recall = np.divide(tp, support, out=np.zeros(len(LABELS)), where=support != 0)
    denominator = support + matrix.sum(axis=0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(LABELS)), where=denominator != 0)
    return {"macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "per_class_recall": dict(zip(LABELS, recall.tolist())),
            "confusion_matrix": matrix.tolist()}


def paired_interval(truth: np.ndarray, first: np.ndarray, second: np.ndarray) -> list[float]:
    rng = np.random.default_rng(42)
    strata = [np.flatnonzero(truth == label) for label in range(len(LABELS))]
    differences = []
    # One clip per source group: resample independent sources, never audio frames.
    for _ in range(2000):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        differences.append(metrics(truth[indices], second[indices])["macro_f1"]
                           - metrics(truth[indices], first[indices])["macro_f1"])
    return np.quantile(differences, [0.025, 0.975]).tolist()


def extract(clips, vad) -> tuple[np.ndarray, np.ndarray, dict]:
    mel, extra, elapsed = [], [], {"mel_seconds": 0.0, "vad_seconds": 0.0, "audio_seconds": 0.0}
    for index, clip in enumerate(clips, 1):
        audio, digest = read_wav(clip.path)
        if digest != clip.pcm_sha256:
            raise ValueError(f"Clip {index}: content changed after validation")
        # Both arms see exactly the same samples; discard at most 1023 tail samples.
        audio = audio[:len(audio) // 1024 * 1024]
        start = time.perf_counter()
        mel.append(mel_features(audio))
        elapsed["mel_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        extra.append(vad_features(vad.probabilities(audio)))
        elapsed["vad_seconds"] += time.perf_counter() - start
        elapsed["audio_seconds"] += len(audio) / SAMPLE_RATE
    return np.array(mel), np.array(extra), elapsed


def evaluate(clips, mel: np.ndarray, extra: np.ndarray) -> dict:
    train = np.array([clip.split == "train" for clip in clips])
    labels = np.array([clip.label for clip in clips])
    predictions, results = [], {}
    for name, features in (("mel_rms", mel), ("mel_rms_vad", np.column_stack((mel, extra)))):
        predicted = predict(features[train], labels[train], features[~train])
        predictions.append(predicted)
        results[name] = {"dimensions": features.shape[1], **metrics(labels[~train], predicted)}
    return {"arms": results, "delta_macro_f1": results["mel_rms_vad"]["macro_f1"]
            - results["mel_rms"]["macro_f1"],
            "delta_macro_f1_ci95": paired_interval(labels[~train], *predictions)}


def make_report(kind, clips, vad, mel, extra, elapsed) -> dict:
    counts = Counter((clip.split, clip.label) for clip in clips)
    identity = sorted((clip.pcm_sha256, clip.label, clip.split, clip.group) for clip in clips)
    source_paths = [Path(__file__), Path(__file__).with_name("audio_corpus.py"),
                    Path(__file__).with_name("audio_features.py"),
                    Path(__file__).parents[1] / "adapters/mac_desktop/sensor_mic.py"]
    return {
        "schema": "brain.audio-vad-experiment.v1", "corpus_kind": kind,
        "evidence": "synthetic_smoke_only" if kind == "synthetic" else "exploratory_offline_feature_probe",
        "not_evidence_for": ["SNN improvement", "flow", "stress", "interruptibility", "live microphone accuracy"],
        "labels": LABELS, "confusion_matrix_axes": "rows=true, columns=predicted",
        "counts": {split: {name: counts[split, i] for i, name in enumerate(LABELS)} for split in ("train", "test")},
        "corpus_sha256": hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in source_paths)).hexdigest(),
        "model_sha256": vad.sha256, "pretrained_vad": True,
        "probe": "train-standardized nearest centroid; train-constant features excluded; label-order ties",
        "bootstrap": {"seed": 42, "resamples": 2000, "unit": "independent source group", "stratified_by": "class"},
        "small_sample_warning": any(counts["test", i] < 20 for i in range(len(LABELS))),
        "feature_settings": {"sample_rate": SAMPLE_RATE, "mel_frame": 1024, "vad_frame": 512,
                             "vad_threshold": 0.5, "tail": "discarded equally in both arms"},
        "timing": {**elapsed, "vad_realtime_factor": elapsed["vad_seconds"] / elapsed["audio_seconds"],
                   "scope": "feature extraction wall time, includes first inference; excludes model loading and file IO; not energy"},
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "numpy": np.__version__, "onnxruntime": vad.runtime_version},
        **evaluate(clips, mel, extra),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Explicit local corpus manifest (no downloads)")
    parser.add_argument("--model", type=Path, required=True, help="Trusted local Silero ONNX model")
    parser.add_argument("--model-sha256", required=True, help="Expected model hash, verified before loading")
    parser.add_argument("--output", type=Path, default=Path("benchmark/reports/audio_vad.json"))
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new report filename")
        kind, clips = load_corpus(args.manifest)
        vad = LocalVad(args.model, args.model_sha256)
        mel, extra, elapsed = extract(clips, vad)
        report = make_report(kind, clips, vad, mel, extra, elapsed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except ValueError as exc:
        parser.exit(2, f"Audio experiment: {exc}\n")
    except OSError:
        parser.exit(2, "Audio experiment: local file operation failed; nothing was uploaded\n")
    print(f"Offline feature probe complete ({kind}); no sensors activated. Report written locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
