# Audio feature experiment: Mel/RMS vs. Mel/RMS + VAD

Status: offline benchmark implemented; no measured benefit on recorded audio yet.
This is an exploratory **feature probe**, not an SNN benchmark or a live sensor.
The Observatory and its capture controls are unchanged.

## Question and scope

Does a frozen, locally executed voice activity detector add useful information
to the existing audio features on independently held-out recordings?

VAD detects speech activity, not meeting participation, music, stress or flow.
Podcasts and singing are important counterexamples to “speech means meeting”.
Even a positive result here would only justify a subsequent SNN experiment.
[Silero's documented task and local runtime](https://github.com/snakers4/silero-vad).

The existing `MicSensor._encode` supplies 32 log-compressed Mel bands and RMS
loudness. The benchmark reuses it with `mock_mode=True` and never calls the
sensor's `sample()` method or its synthetic-noise fallback. Graphify's existing
server graph helped separate this experiment from the legacy voice/capture
endpoints; the actual encoder was inspected directly.

## Fixed experiment

| Item | Choice |
| --- | --- |
| Classes | `meeting`, `podcast`, `instrumental_music`, `vocal_music`, `ambient` |
| Arm A | Mean and standard deviation of 32 Mel bands + RMS: 66 features |
| Arm B | Identical 66 features + VAD mean, standard deviation, active fraction and transitions/second: 70 features |
| Probe | Same nearest-centroid classifier; normalization learned only on training data |
| Constant features | Features constant in training are excluded in both arms |
| VAD threshold | 0.5, fixed before looking at test results; no segmentation/hangover postprocessing |
| Main metric | Paired change in macro-F1, Arm B minus Arm A |
| Diagnostics | Balanced accuracy, class recall, full confusion matrices, extraction wall time |
| Uncertainty | 2,000 paired, class-stratified source-group bootstrap samples; seed 42 |
| Model | Frozen Silero VAD v6.0 ONNX, CPU, one inference thread |

The probe is supervised; it is not the production SNN. Silero is pretrained, so
the augmented pipeline must not be described as entirely free of pretraining.
Its additional information comes from externally learned speech features.

Both arms receive the same waveform samples. Mel uses consecutive 1,024-sample
windows; VAD uses 512-sample windows with its required 64-sample context. Less
than 64 ms at a clip's end is discarded equally in both arms. Recurrent VAD state
and context are reset for every clip. Features are summarized over the whole
clip: this does **not** measure online detection latency or causal streaming behavior.
[Pinned model interface](https://github.com/snakers4/silero-vad/blob/v6.0/src/silero_vad/utils_vad.py).

The interval describes held-out source sampling, conditional on this fixed
training set. It does not cover training-set variability, annotation errors or
device/domain shift. It is particularly unstable with very few sources.

## Corpus contract

Supply an explicit local JSON manifest and existing WAV files. The runner never
records, downloads or transcribes audio. Do not use private meetings without
permission from the people whose speech is present. Prefer appropriately
licensed test material or deliberately consented research recordings. A
`license` entry documents the provider's declaration; the program cannot verify
that declaration or independently establish permission.

- WAV: mono, 16-bit PCM, 16 kHz, 2–30 seconds. No silent resampling or downmixing.
- Each source group is used **once in the entire corpus**. No train/test reuse
  of a speaker, recording, podcast series, track/artist or transformed copy.
  Merge overlapping identities into one group; choose one representative clip.
- The same PCM waveform under another filename is rejected. Different excerpts
  or transformed copies cannot be reliably detected automatically: grouping
  remains the corpus curator's responsibility.
- Freeze the manifest before evaluation. Match capture conditions, duration,
  loudness and background-noise distributions across classes and splits.
  Otherwise a classifier may learn recording equipment or corpus identity.
- `ambient`: non-speech/non-music background sounds. `meeting`: known
  conversational source context, not a label inferred by VAD. Mark ambiguous or
  mixed material separately in corpus preparation; document exclusions before
  testing and do not remove errors retrospectively.
- The CLI minimum is 20 clips: 2 per class in each split. That is a plumbing
  minimum, **not** adequate evidence. Aim initially for at least 20 independent
  sources per class per split (200 total); sample size still needs justification.
- `kind: "synthetic"` always produces `synthetic_smoke_only` evidence. Declaring
  `recorded` does not authenticate the data or turn a result into a product claim.

Example manifest structure; the two illustrative entries alone are incomplete:

```json
{
  "schema": "brain.audio-corpus.v1",
  "kind": "recorded",
  "clips": [
    {
      "path": "m01.wav",
      "label": "meeting",
      "split": "train",
      "group": "m01",
      "license": "Document the actual permission or license here"
    },
    {
      "path": "v01.wav",
      "label": "vocal_music",
      "split": "test",
      "group": "v01",
      "license": "Document the actual permission or license here"
    }
  ]
}
```

Use opaque identifiers, not names. Relative WAV paths must resolve inside the
manifest directory, including through symlinks. `benchmark/audio-data/` is
ignored by Git; this is an accidental-commit guard, not encryption or deletion.
Put the corpus there or outside the repository. Keep original permissions and
provenance locally, separate from aggregate benchmark reports.

## Run locally

Run from the repository root using the existing project environment. The only
new optional runtime is `onnxruntime==1.24.4`, listed under `audio-benchmark`.
If missing, install the project's optional dependency set in your environment.
Do not install a new inference stack just to run the default unit tests.

Obtain the model separately from the
[official pinned v6.0 model directory](https://github.com/snakers4/silero-vad/tree/v6.0/src/silero_vad/data)
and retain its [MIT license](https://github.com/snakers4/silero-vad/blob/v6.0/LICENSE)
when redistributing it. The runner does not fetch models or execute Torch Hub code.

Verified SHA-256 of `silero_vad.onnx` downloaded from that source on 2026-09-14:

```text
597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004
```

This is our recorded download fingerprint, not an independently signed upstream
checksum. Model bytes are hash-checked before ONNX loading; a mismatch aborts.
Do not load a model from an untrusted contributor simply because they supply a hash.

```sh
rtk proxy .venv/bin/python -m benchmark.audio_vad \
  benchmark/audio-data/manifest.json \
  --model models/silero-v6.0.onnx \
  --model-sha256 597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004 \
  --output benchmark/reports/audio-vad-run-001.json
```

Existing report files are never overwritten. Reports contain aggregate metrics,
counts, environment versions, model/corpus/source fingerprints and timing—not
WAV paths, group IDs, waveforms, transcripts, per-clip features or predictions.
In-memory feature extraction is not a claim that audio features are anonymous
or that RAM is securely erased. Existing input WAV files are left untouched.

Runtime failures abort rather than fabricate VAD scores. No microphone, system
audio loopback, network client, desktop sensor, live model or checkpoint is used.
ONNX Runtime telemetry is disabled. Wall time is not energy consumption.

## Decision after recorded-data evaluation

Proposed go/no-go gate for **a further experiment**, fixed before that run:
at least +0.05 absolute macro-F1, a paired interval above zero, no class losing
more than 0.05 recall, and added VAD extraction time below 0.1× audio duration on
the target Mac. These are engineering targets, not established scientific cutoffs.
Review small samples and confounds even when all targets are met. Report negative
results with the same detail; do not repeatedly tune against the held-out set.

Then compare SNN Mel vs. SNN Mel+VAD with matched input drive, training exposure,
random seeds and a classical baseline. Hold all desktop inputs constant for the
audio-only ablation. Add app/activity context separately to measure its contribution.
No SNN gain is established by this feature probe alone.

Integration caveat from the current core: the separate `ConceptTracker` masks
Mel channels (`tracker_spikes[112:145] = 0`), whereas the SNN concept pathway is
different. Explicitly identify which representation the next SNN evaluation
measures; this offline probe does not establish an improvement in that tracker.

Before any live audio integration: separate explicit opt-in for the actual source
(microphone versus system playback), a working stop control at capture level,
no retained raw audio by default, no upload, and clear status/error reporting.
A microphone cannot be assumed to hear playback through headphones. Test that
domain separately; file-based results do not establish microphone performance.

## Contributor task

First contribution: propose a reproducible, legally usable corpus manifest with
source-group separation and all five classes, then run both feature arms and
report failures as well as gains. No cloud accounts or hardware purchase needed.
Do not upload private recordings with an issue or pull request.

## Verification

```sh
rtk proxy .venv/bin/python -m pytest tests/test_audio_vad.py -q
```

The real-model test is opt-in, uses only generated tones/silence and never fetches
assets. Set `BRAIN_VAD_TEST_MODEL` to a trusted local ONNX file and
`BRAIN_VAD_TEST_SHA256` to the fingerprint above to include it. Without these,
that one test is skipped. No synthetic accuracy number should be presented as
speech/music evidence. Capture-control regression tests remain independent.
