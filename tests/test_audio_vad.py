"""Synthetic plumbing tests, not speech/music accuracy evidence."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from adapters.mac_desktop.sensor_mic import MicSensor
from benchmark import audio_vad
from benchmark.audio_corpus import LABELS, load_corpus, read_wav
from benchmark.audio_features import LocalVad, mel_features, vad_features


def write_wav(path, samples, rate=16000, channels=1):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())


@pytest.fixture(autouse=True)
def no_capture_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline experiment attempted device or network access")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=forbidden))
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def corpus(tmp_path):
    rows = []
    t = np.arange(32000) / 16000
    for label in LABELS:
        for split in ("train", "test"):
            for _ in range(2):
                index = len(rows)
                path = f"clip-{index}.wav"
                samples = 4000 * np.sin(2 * np.pi * (200 + 40 * index) * t)
                write_wav(tmp_path / path, samples)
                rows.append({"path": path, "label": label, "split": split,
                             "group": f"source-{index}", "license": "Self-generated test tone"})
    data = {"schema": "brain.audio-corpus.v1", "kind": "synthetic", "clips": rows}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest, data


def test_corpus_loads_explicit_local_files(corpus):
    kind, clips = load_corpus(corpus[0])
    assert kind == "synthetic" and len(clips) == 20
    assert len({clip.group for clip in clips}) == 20
    assert len({clip.pcm_sha256 for clip in clips}) == 20


@pytest.mark.parametrize("field,value", [
    ("label", "flow"), ("split", "validation"), ("license", ""),
    ("group", []), ("path", "/private-not-allowed.wav"),
    ("path", "../outside.wav"), ("extra", True),
])
def test_manifest_rejects_invalid_inputs(corpus, field, value):
    manifest, data = corpus
    data["clips"][0][field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Clip 1") as error:
        load_corpus(manifest)
    assert "private-not-allowed" not in str(error.value)


def test_group_cannot_cross_splits(corpus):
    manifest, data = corpus
    data["clips"][2]["group"] = data["clips"][0]["group"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="independent source"):
        load_corpus(manifest)


def test_duplicate_waveform_is_rejected_despite_new_filename(corpus):
    manifest, data = corpus
    first = manifest.parent / data["clips"][0]["path"]
    second = manifest.parent / data["clips"][2]["path"]
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="Duplicate PCM"):
        load_corpus(manifest)


def test_symlink_cannot_escape_corpus(corpus, tmp_path):
    manifest, data = corpus
    nested = tmp_path / "restricted"
    nested.mkdir()
    (nested / "linked.wav").symlink_to(tmp_path / "clip-0.wav")
    data["clips"][0]["path"] = "linked.wav"
    new_manifest = nested / "manifest.json"
    new_manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside the manifest"):
        load_corpus(new_manifest)


def test_every_class_must_exist_in_both_splits(corpus):
    manifest, data = corpus
    data["clips"][2]["split"] = "train"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="train AND test"):
        load_corpus(manifest)


@pytest.mark.parametrize("rate,channels,size", [(8000, 1, 32000), (16000, 2, 64000), (16000, 1, 16000)])
def test_wav_format_is_explicit(tmp_path, rate, channels, size):
    path = tmp_path / "invalid.wav"
    write_wav(path, np.zeros(size), rate=rate, channels=channels)
    with pytest.raises(ValueError):
        read_wav(path)


def test_truncated_wav_is_rejected(tmp_path):
    path = tmp_path / "truncated.wav"
    write_wav(path, np.zeros(32000))
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="Truncated"):
        read_wav(path)


def test_mel_features_reuse_existing_encoder_without_device():
    audio = np.linspace(-0.2, 0.3, 2048, dtype=np.float32)
    sensor = MicSensor(mock_mode=True)
    values = [sensor._encode(chunk) for chunk in audio.reshape(-1, 1024)]
    expected = np.array([[*row["mel"], row["rms"]] for row in values])
    actual = mel_features(audio)
    assert actual.shape == (66,)
    np.testing.assert_allclose(actual, np.r_[expected.mean(axis=0), expected.std(axis=0)])


def test_vad_feature_semantics():
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    features = vad_features(scores)
    np.testing.assert_allclose(features, [0.5, scores.std(), 0.5, 2 / 0.128])


def test_vad_context_and_state_reset_per_clip():
    calls = []
    def run(_, inputs):
        calls.append({key: value.copy() for key, value in inputs.items()})
        return np.array([[0.25]], dtype=np.float32), np.ones((2, 1, 128), dtype=np.float32)
    vad = LocalVad.__new__(LocalVad)
    vad.session = SimpleNamespace(run=run)
    audio = np.full(1024, 0.2, dtype=np.float32)
    for _ in range(2):
        np.testing.assert_array_equal(vad.probabilities(audio), [0.25, 0.25])
    for index in (0, 2):
        assert not calls[index]["state"].any()
        assert not calls[index]["input"][:, :64].any()
    assert calls[1]["state"].all()
    np.testing.assert_allclose(calls[1]["input"][:, :64], 0.2)


@pytest.mark.parametrize("score", [float("nan"), -0.1, 1.1])
def test_invalid_vad_score_never_becomes_a_fake_observation(score):
    vad = LocalVad.__new__(LocalVad)
    vad.session = SimpleNamespace(run=lambda *_: (np.array([[score]]), np.zeros((2, 1, 128))))
    with pytest.raises(ValueError, match="invalid score"):
        vad.probabilities(np.zeros(1024, dtype=np.float32))


def test_model_hash_checked_before_runtime_load(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not-a-model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        LocalVad(path, "0" * 64)


def test_predict_uses_training_statistics_only():
    labels = np.repeat(np.arange(5), 2)
    train = np.column_stack((labels.astype(float), np.ones(10)))
    test = np.array([[1.1, 1e9], [1e8, -1e9]])
    np.testing.assert_array_equal(audio_vad.predict(train, labels, test), [1, 4])
    assert audio_vad.predict(train, labels, test[:1])[0] == 1


def test_known_metrics_and_paired_interval():
    truth = np.tile(np.arange(5), 2)
    result = audio_vad.metrics(truth, truth)
    assert result["macro_f1"] == result["balanced_accuracy"] == 1.0
    assert audio_vad.metrics(truth, (truth + 1) % 5)["macro_f1"] == 0.0
    assert audio_vad.paired_interval(truth, truth, truth) == [0.0, 0.0]


class StubVad:
    sha256 = "a" * 64
    runtime_version = "test-double"

    def __init__(self, *args):
        pass

    def probabilities(self, audio):
        return np.full(len(audio) // 512, 0.2)


def test_cli_report_contains_no_clip_paths_or_features(corpus, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audio_vad, "LocalVad", StubVad)
    output = tmp_path / "report.json"
    args = [str(corpus[0]), "--model", "unused", "--model-sha256", "a" * 64, "--output", str(output)]
    assert audio_vad.main(args) == 0
    raw = output.read_text()
    report = json.loads(raw)
    assert report["evidence"] == "synthetic_smoke_only"
    assert report["small_sample_warning"] is True
    assert report["arms"]["mel_rms"]["dimensions"] == 66
    assert report["arms"]["mel_rms_vad"]["dimensions"] == 70
    assert all(value not in raw for value in (str(tmp_path), "clip-0.wav", "source-0", "Self-generated"))
    assert "features\":" not in raw
    assert "clip-0" not in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        audio_vad.main(args)
    assert error.value.code == 2
    assert output.read_text() == raw


def test_changed_corpus_is_rejected(corpus):
    _, clips = load_corpus(corpus[0])
    write_wav(clips[0].path, np.ones(32000))
    with pytest.raises(ValueError, match="changed after validation"):
        audio_vad.extract(clips, StubVad())


@pytest.mark.skipif(not os.environ.get("BRAIN_VAD_TEST_MODEL"), reason="Explicit local model required; never downloaded by tests")
def test_real_onnx_end_to_end(corpus, tmp_path):
    model = Path(os.environ["BRAIN_VAD_TEST_MODEL"])
    digest = os.environ["BRAIN_VAD_TEST_SHA256"]
    vad = LocalVad(model, digest)
    silence = np.zeros(2048, dtype=np.float32)
    first = vad.probabilities(silence)
    vad.probabilities(np.full(2048, 0.1, dtype=np.float32))
    np.testing.assert_allclose(vad.probabilities(silence), first, atol=1e-7)
    output = tmp_path / "real-model-synthetic-audio.json"
    audio_vad.main([str(corpus[0]), "--model", str(model), "--model-sha256", digest, "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert report["evidence"] == "synthetic_smoke_only"
