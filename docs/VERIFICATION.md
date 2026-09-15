# Verification and known gaps

Documentation review: **2026-09-15**. Runtime code was not changed in this pass.
These checks validate selected contracts, not product readiness or scientific claims.

## Current results

| Check | Result |
| --- | --- |
| Capture, telemetry, audio pipeline, brain tools, HTTP/MCP, experience and episode tests | 85 passed, 1 skipped |
| Observatory, live/capture UI contracts and bilingual tour | 37 passed |
| 3D structural checks | PASS: 1,000 units, shared Circuit palette, atlas placement and asset consistency |
| Known macOS control test | 1 failed: this Python lacks `os.waitid` |
| Fresh installation satisfying current dependency constraints | Not verified |
| Real sensor, wearable, cloud model, voice recording or personal checkpoint run | Not performed |
| Longitudinal study or real speech/music accuracy evaluation | Not performed |

Environment: Python package versions PyTorch **2.11.0**, snnTorch **0.9.4**,
MCP **2.2.0**, pytest **9.0.3**. The existing environment does **not** meet the
current `torch>=2.13` requirement. Passing tests here do not verify that new floor.
No dependency downgrade or environment migration was performed.

## Reproduce the scoped checks

From the repository root, in a prepared environment:

```sh
.venv/bin/python -m pytest -q --override-ini addopts= \
  tests/test_capture_controls.py tests/test_observation_telemetry.py \
  tests/test_audio_vad.py tests/test_brain_tools.py tests/test_tools_endpoint.py \
  tests/test_mcp_server.py tests/test_experience.py tests/test_episode_log.py

node --test docs/dashboard-concepts/verify-onboarding.mjs \
  docs/dashboard-concepts/verify-observatory.mjs \
  docs/dashboard-concepts/verify-live.mjs \
  docs/dashboard-concepts/verify-capture.mjs

node 3d/verify-brain.mjs
node scripts/check-docs.mjs
git diff --check
```

The optional real ONNX test is skipped without an explicitly supplied trusted
model and fingerprint. It must not download or record audio implicitly.
See the [audio experiment](experiments/audio-vad.md).

The tour's separate 2026-09-15 browser check covered all 18 steps in both
languages, reload persistence and narrow layouts. No new visual redesign was
performed in this documentation pass. Historical visual checks are recorded in
[EINFUEHRUNG.md](dashboard-concepts/EINFUEHRUNG.md); they do not certify every
browser, screen reader or hardware GPU.

## Tests requiring care

Do not run the full unfiltered suite or `scripts/run_tests.sh` as a promise of
sensor-free execution. In particular, these constructors can touch native APIs:

- `tests/test_sensor_keymouse.py::test_keystroke_construction`
- `tests/test_sensor_keymouse.py::test_mouse_construction`
- `tests/test_sensor_mic.py::test_construction`

`tests/test_control.py::test_running_pid_reaps_a_crashed_child` currently fails
on this macOS interpreter at `os.waitid`, before its intended assertion. It was
reproduced separately; it is not counted among passing checks. Cross-platform
test repair and clean-environment validation remain engineering work.

## Scientific evidence boundary

`benchmark/run_all.py` currently runs four synthetic diagnostic groups:
stability, replay, discrimination and modulator reactivity. Its success text is
not a scientific validation claim. The README no longer describes a fifth
sleep-retention benchmark as part of that runner.

Do not present historical scores, target thresholds or a successful test harness
as real-world focus/stress detection. Before reporting benefit, freeze an
evaluation protocol with baselines, held-out sessions, provenance and failures.
