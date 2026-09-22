# Running brAIn

Run commands from the repository root. Do not run real capture as part of a
documentation check. [Privacy](PRIVACY.md) · [Verification](VERIFICATION.md)

## Static dashboard

```sh
python3 -m http.server 4178 --bind 127.0.0.1 --directory docs/dashboard-concepts
```

Open http://127.0.0.1:4178/observatory.html. It includes five synthetic exploration
pages plus the separate **Live session** page. **Introduction** opens an 18-step
English/German tour. No Node package installation is needed to serve this UI.

## Python environment

```sh
uv venv --python 3.12
uv pip install -e ".[dev]"
uv pip check
```

Python must satisfy `>=3.11,<3.13`. The manifest requires `torch>=2.13`.
A separate macOS/Python 3.11.13 environment with `torch==2.13.0` was installed
and dependency-checked on 2026-09-17. See [test scope](VERIFICATION.md).
This does not automatically upgrade an existing environment: the maintainer's
older `.venv` still has 2.11.0 and was deliberately left untouched.
Do not downgrade requirements or alter a running research environment silently.

## Opt-in observer: recommended first runtime

```sh
# Status-only; no source can be enabled.
.venv/bin/python -m server.observe

# Alternatively, allow explicit capture choices; all four start off.
.venv/bin/python -m server.observe --desktop-metadata

# Optional bounded session.
.venv/bin/python -m server.observe --desktop-metadata --duration 120
```

Run one variant at a time. The observer binds to **127.0.0.1:8001**.
In Observatory, select sources under **Data & privacy**, then connect the view.
Available sources: keyboard event rate, pointer event rate, idle time and coarse
foreground-app category. Input Monitoring availability is reported separately
from whether a source was selected. No microphone, screenshots or wearables.

A fresh seeded model is used. All-off pauses model steps. Stopping capture does
not erase already-derived weights or retained frames. Ctrl-C ends the process;
the observer does not save a checkpoint.

## Persistent daemon: separate experimental runtime

Read the [data inventory](PRIVACY.md) first. The daemon captures nothing until
you share a source under **Data sources** in the training console: keyboard
rhythm, pointer activity, idle time, apps and microphone features, each on its
own. The choice is saved in `checkpoints/consent.json` and restored on restart;
a missing or unreadable file means nothing is shared, and a source added in a
later version starts off. While nothing is shared the model does not step.
Switching a source off stops its listener, stream or polling at once and
removes its last value. Observatory capture switches do not control this process.

```sh
.venv/bin/python -m server.control
```

Open http://127.0.0.1:8900. The control server serves `train-ui/index.html`
and starts/stops a daemon on **127.0.0.1:8000**. It does not automatically start a
new daemon unless requested, but supervises an existing desired-running pidfile.

The daemon loads or creates `checkpoints/braind.sqlite`, saves periodically
and on normal shutdown, and maintains companion databases in the same directory.
Ollama is optional for learning; chat/proactive language output requires a working
backend. `config.json` currently disables the cloud route. Not every historical
sensor/architecture config field is wired into runtime control.

### Isolated synthetic daemon run

```sh
# The temporary directory keeps test persistence separate from personal state.
brain_trial_dir=$(mktemp -d)
.venv/bin/python -m server.braind start --mock-sensors --port 8010 \
  --checkpoint "$brain_trial_dir/brain.sqlite"
```

Mock mode replaces desktop acquisition with synthetic values, but sources
still start off. Its choice is kept in `consent-mock.json`, so agreeing to
synthetic data never switches on real capture later. Share them to get data:

```sh
curl -s http://127.0.0.1:8010/api/consent        # shows the current revision
curl -s -X POST http://127.0.0.1:8010/api/consent \
  -H 'Content-Type: application/json' \
  -d '{"revision": 0, "enabled": {"idle": true, "active_app": true}}'
```

Voice recording is refused unless the microphone source is shared, and always
in mock mode. Mock mode does not disable network-capable tools or the optional
LLM path; do not invoke them in a sensor-free test. Stop with Ctrl-C.
Do not connect the standard training console to this custom port without
checking its configuration; this command is intended for isolated API testing.

### Erase the daemon's data

```sh
.venv/bin/python -m server.braind erase          # lists what would be deleted
.venv/bin/python -m server.braind erase --yes    # deletes it
```

Stop the daemon in the console first; the command refuses while it runs.
It covers the stores in the [data inventory](PRIVACY.md#persistent-files),
their SQLite side files, the daily backups and the consent choice. Add
`--checkpoint PATH` for a non-default location. The next start is a fresh
brain that shares nothing.

### Optional login service

`scripts/install_launchd.sh` installs an always-on control server that can
autostart the daemon and restart it after crashes. This is an explicit opt-in
to background operation, not part of the default quick start. Inspect its output
and the required macOS permissions before enabling it.
`scripts/uninstall_launchd.sh` removes the service and stops its managed daemon.

Input Monitoring and Microphone permissions apply to the actual launching
Python/terminal identity. Never grant either solely to get a test suite green.

## MCP

See [MCP.md](MCP.md). The proxy reads the persistent daemon, not the observer.
Pointing it at port 8001 does not make Observatory a persistent context service.

## Removed and unsupported paths

- `pet-face/` and its Tauri build were removed; do not run old `npx tauri dev`
  instructions. Voice backend code remains.
- No smartwatch connector, screen-vision input or hardware-companion firmware is
  provided as a supported installation path.
- Keep services on loopback. Do not expose them through a public reverse proxy,
  tunnel or wildcard bind; current local APIs are not authenticated deployment APIs.
