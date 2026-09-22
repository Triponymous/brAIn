# brAIn — Living Desktop Manager

A local-first research project exploring **personal context learned over time**:
a small spiking neural network, an inspectable dashboard, and an optional
context interface for AI assistants.

The long-term idea is a desktop companion that learns useful patterns and helps
at appropriate moments. The current repository is a research prototype, not a
validated emotion detector, autonomous desktop manager or production security boundary.

[Get started](#quick-start) · [Documentation](docs/README.md) ·
[Architecture](docs/living-desktop-manager.md) · [Roadmap](docs/ROADMAP.md) ·
[Privacy](docs/PRIVACY.md) · [Contributing](CONTRIBUTING.md)

## What works today

| Component | Implemented | Important boundary |
| --- | --- | --- |
| SNN core | PyTorch LIF steps, fixed expansion, STDP/BCM, competitive concepts, working memory and model modulators | Online learning is implemented; real-world usefulness still needs controlled evaluation |
| Observatory | Interactive 3D atlas, Circuit view, inspectors, timelines and an 18-step English/German introduction | Five pages use synthetic demonstration data; only **Live session** reads observed model steps |
| Opt-in observation | Four individually switchable desktop-metadata sources, stop control, captured frames and JSON export | Separate fresh model; no persisted checkpoint, microphone, LLM or wearable access |
| Research daemon | Persistent model, learned labels, episode/experience logs, training console and optional LLM/voice paths | Broader data access than Observatory: five sources, each off until shared in the training console; not controlled by Observatory switches |
| MCP context interface | Nine query tools over stdio, backed by the daemon | Reads can expose personal metadata to the client and are logged locally |
| Audio experiment | Offline Mel/RMS versus Mel/RMS + VAD evaluation pipeline | No established improvement on real recordings; not integrated into live capture |

“Living” describes the research direction. DA, NE, ACh and 5HT are **model
variables**, not measurements of human neurotransmitters. Labels such as “flow”
are user-taught associations, not independently verified psychological states.
There is no demonstrated superiority to simple baselines or evidence of consciousness.

## Quick start

### 1. Explore the dashboard without collecting data

Python 3 is sufficient for the static dashboard; no model or sensor permissions:

```sh
git clone https://github.com/Triponymous/brAIn.git
cd brAIn
python3 -m http.server 4178 --bind 127.0.0.1 --directory docs/dashboard-concepts
```

Open [Observatory](http://127.0.0.1:4178/observatory.html).
Choose **Introduction** for the guided tour. English is the default; Deutsch
remains available, including during a tour. The chosen language is stored locally.

Loading the page does not enable capture. If an observation runner is already
running, the dashboard can display its existing capture status; closing a tab
does not stop that runner.

### 2. Install the research environment

The package declares Python **3.11 or 3.12**. Real desktop sensors target macOS;
mock/offline tests do not establish cross-platform sensor support.

```sh
uv venv --python 3.12
uv pip install -e ".[dev]"
uv pip check
```

Dependency constraints are in [pyproject.toml](pyproject.toml), including
`torch>=2.13`. Do not silently lower that requirement to make installation pass.
A separate macOS/Python 3.11 environment with PyTorch 2.13.0 was installed and
dependency-checked on 2026-09-17; see [verification scope](docs/VERIFICATION.md).

### 3. Optionally observe real model steps

In a second terminal, from the repository root:

```sh
.venv/bin/python -m server.observe --desktop-metadata
```

Open **Live session → Data & privacy**. All four sources begin **off** in a new
runner session. Enable only sources you want, then use **Connect local model**.

- **Stop all capture** stops source queries and subsequent model steps.
- **Disconnect view** or freezing the timeline does not stop capture.
- **Ctrl-C** stops the observation runner.
- No checkpoint is loaded or saved. Exports remain your responsibility.

Without `--desktop-metadata`, the runner provides connection/status inspection
but does not allow sources to be enabled. See [runtime guide](docs/RUNNING.md)
and [capture details](docs/dashboard-concepts/LIVE-DATA.md).

## Persistent research and AI context

For persistent learning, use the separate training console and daemon described
in [Running brAIn](docs/RUNNING.md). Review [privacy boundaries](docs/PRIVACY.md)
before sharing sources: once switched on, this path can include microphone
features and app metadata, plus disk persistence, chat and optional voice/cloud
functionality. The daemon captures nothing until a source is shared.

The `brain-mcp` executable exposes nine query tools from that daemon:
state, history, concepts, learned labels, habits, anomalies, explanations,
recall and experience summaries. It does **not** connect to Observatory's
port-8001 observation model.

See [MCP interface](docs/MCP.md) for transport, tools and disclosure limits.
A compatible host may use these tools to ground responses. This does not train
frontier-model weights or guarantee support in every ChatGPT/Claude application.
Client-specific setup and a reusable integration skill remain separate concerns.

## Architecture

The default `Brain()` contains 200 sensory, 500 expansion, 200 concept and
100 working-memory units. Expansion outputs are binary projections, not LIF
membranes. Restored checkpoints may have different dimensions; the runtime's
exported architecture is authoritative, not unused configuration fields.

```text
Desktop inputs → encoding → SNN + concept tracking
                              ├─ observe → Observatory Live session (ephemeral)
                              └─ braind → checkpoint + logs + learned labels
                                            ├─ training console
                                            ├─ optional LLM/voice/capabilities
                                            └─ query API → stdio MCP → chosen client
```

These are separate runtime paths, not two views of one shared model.
The anatomical 3D asset is a visual reference: computational role placement is
illustrative and live synaptic connections are not exported.
[Architecture and limits](docs/living-desktop-manager.md).

## Research priorities

1. Reproducible sessions, explicit consent and a complete data lifecycle.
2. Personal adaptation evaluated against simple rules and classical baselines.
3. A permission-scoped context interface with freshness, provenance and revocation.
4. Offline VAD and sparse-circuit ablations before live integration.
5. Optional wearables, interruption policies and resumption cards only after
   their incremental value can be tested.

These are proposed milestones, not shipped features or promised dates.
[Roadmap and acceptance criteria](docs/ROADMAP.md).

## Contributing and verification

See [CONTRIBUTING.md](CONTRIBUTING.md) for safe tests, branch conventions and PR
expectations. Existing benchmark scripts are research diagnostics; historical
synthetic scores are not validated stress detection, battery savings or
longitudinal user outcomes. [Current verification scope](docs/VERIFICATION.md).

The former Tauri `pet-face/` application has been removed. Voice backend code
still exists; there is no current Pet-Face build target. A hardware companion
and wearable integrations are proposals, not supported products.

## Author and license

Created by **Leon Matthies (Triponymous)**, an independent research project.

Code: [MIT](LICENSE). The derived anatomical asset is separately licensed under
**CC BY-SA 4.0**, with required attribution in
[brain-LICENSE.md](docs/dashboard-concepts/assets/brain-LICENSE.md).
Bundled Three.js retains its [MIT notice](docs/dashboard-concepts/vendor/three/LICENSE.txt).
Do not treat the whole repository's media as MIT or upload personal observations.
