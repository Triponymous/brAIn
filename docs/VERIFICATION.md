# Verification and known gaps

Verification update: **2026-09-17**. Runtime code was not changed; the control
test now exercises child reaping without the unavailable macOS `os.waitid` API.
These checks validate selected contracts, not product readiness or scientific claims.

## Current results

| Check | Result |
| --- | --- |
| Python suite in clean PyTorch-2.13 environment | 682 passed, 1 skipped, 3 native sensor tests deselected |
| Observatory, live/capture UI contracts and bilingual tour | 37 passed |
| 3D structural checks | PASS: 1,000 units, shared Circuit palette, atlas placement and asset consistency |
| macOS control tests | 11 passed after portable test repair |
| Fresh installation satisfying current dependency constraints | Installed with PyTorch 2.13.0; `uv pip check` passed |
| Real sensor, wearable, cloud model, voice recording or personal checkpoint run | Not performed |
| Longitudinal study or real speech/music accuracy evaluation | Not performed |

The earlier 2026-09-15 scoped run (85 passed, 1 skipped) used PyTorch **2.11.0**, snnTorch
**0.9.4**, MCP **2.2.0** and pytest **9.0.3**. That existing `.venv` does not meet
the current `torch>=2.13` requirement and was left untouched.

On 2026-09-17 a separate temporary environment was created with Python **3.11.13**,
PyTorch **2.13.0**, snnTorch **1.0.0**, MCP **2.2.0**, pytest **9.1.1** and
Anthropic SDK **1.6.0**. All 90 installed distributions passed `uv pip check`.
The install used the declared project dependencies with an explicit
`torch==2.13.0` constraint to exercise the minimum, not an in-place downgrade.
Package source: [official PyTorch distribution on PyPI](https://pypi.org/project/torch/).
This verifies this macOS/Python combination, not every supported platform.
The full selected Python run took 77.02 seconds and emitted one upstream
Starlette/AnyIO `BlockingPortal` deprecation warning. There were no test failures.
The 37 JavaScript tests were rerun in the existing environment; their Python
telemetry subprocess still uses the repository's older `.venv`.

Reproduce the isolated installation and selected full suite:

```sh
brain_check_dir=$(mktemp -d)
uv venv --python 3.11 "$brain_check_dir/venv"
uv pip install --python "$brain_check_dir/venv/bin/python" -e ".[dev]" "torch==2.13.0"
uv pip check --python "$brain_check_dir/venv/bin/python"
"$brain_check_dir/venv/bin/python" -m pytest -q --override-ini addopts= \
  --deselect=tests/test_sensor_keymouse.py::test_keystroke_construction \
  --deselect=tests/test_sensor_keymouse.py::test_mouse_construction \
  --deselect=tests/test_sensor_mic.py::test_construction
```

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

`tests/test_control.py::test_running_pid_reaps_a_crashed_child` now lets the
actual `_running_pid()` probe reap its own exited test child, with a bounded
deadline and cleanup on failure. It still asserts that the child was reaped;
the test is not skipped or replaced with a mock. Production supervision logic
and running services were not changed.

## Scientific evidence boundary

`benchmark/run_all.py` currently runs four synthetic diagnostic groups:
stability, replay, discrimination and modulator reactivity. Its success text is
not a scientific validation claim. The README no longer describes a fifth
sleep-retention benchmark as part of that runner.

Do not present historical scores, target thresholds or a successful test harness
as real-world focus/stress detection. Before reporting benefit, freeze an
evaluation protocol with baselines, held-out sessions, provenance and failures.
