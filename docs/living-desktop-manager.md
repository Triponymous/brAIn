# Living Desktop Manager: direction and architecture

Status reviewed against source on **2026-09-15**.
[Documentation index](README.md) · [Roadmap](ROADMAP.md) · [Privacy](PRIVACY.md)

## Research thesis

Can a small, continuously adapting local model provide useful personal context
to a desktop interface and to a user's chosen AI assistant?

The SNN learns associations through local plasticity; an optional LLM interprets
available context. This separation does not make the system biologically alive,
prove emotional understanding or update the LLM's weights. “Living Desktop
Manager” is the intended direction, not an existing autonomous product.

## Two independent runtimes

| | Opt-in observer | Persistent daemon |
| --- | --- | --- |
| Entry point | `server.observe` | `server.braind start` |
| HTTP | 127.0.0.1:8001 | 127.0.0.1:8000 |
| Interface | Observatory Live session | Training console via control server on 8900 |
| Model | Fresh seeded instance | Loaded checkpoint or fresh model |
| Capture | Four sources, off by default | Broader desktop sensors including microphone features |
| Storage | Bounded RAM window, explicit export | Checkpoints, episodes, experience and grants |
| External model | None | Optional internal LLM; external stdio MCP proxy |
| Shared controls | None | Observatory stop switches do not govern this daemon |

The five other Observatory pages use deterministic synthetic demonstration data.
A live tick is not a demo frame, a recording session or an entire saved brain.

## Model and representations

`brain/core.py` defaults to 200 sensory, 500 expansion, 200 concept and 100
working-memory units. The expansion is a fixed sparse projection; its binary
output has no membrane potential. LIF, competitive concept dynamics, STDP/BCM,
recurrent memory and model modulators implement the learning experiment.

`ConceptTracker` is a separate clustering representation, not the SNN concept
layer. The current tracker masks Mel channels. An audio evaluation must state
which representation it measures rather than attributing tracker outcomes
automatically to SNN synapses.

DA, NE, ACh and 5HT derive from model prediction-error dynamics. They are not
human biochemical measurements. Sleep/consolidation is a model regime, not
evidence about biological sleep or clinical benefit.

## Labels, history and experience

- `bridge/felt_state.py` stores user-taught prototypes associated with model
  signatures and behavioural clusters. Confidence is not calibrated clinical
  certainty. The proactive watcher can freeze a signature for later correction.
- `bridge/episode_log.py` persists episode summaries and prunes rows older than
  90 days by default. App information and labels may be personal data.
- `bridge/experience.py` records events and response links, then settles a
  consequence signature approximately 120 seconds later. Its default does not
  implement the episode log's 90-day retention policy.
- `server/tools.py` records external query names, arguments and provenance in
  the experience log. “Read-only” refers to model/action access, not zero disk writes.

A later signature is an observational outcome, not causal proof that an action
helped. An experience count alone does not establish adaptation or user benefit.

## Context for AI assistants

`bridge/brain_tools.py` defines nine query tools.
`bridge/llm_local.py` and `bridge/llm_cloud.py` implement tool-result loops for
the daemon's optional chat path. `server/mcp.py` separately proxies the query API
over stdio for compatible external hosts. [MCP contract](MCP.md).

SCP (`bridge/scp_*.py`) is the project's internal protocol and includes action
paths. It is not the same thing as MCP and must not be described as read-only
just because the external MCP adapter is restricted to queries.

The current external interface lacks per-client field consent, expiry controls,
pairing and an authenticated remote transport. It must remain local and trusted.
A production context layer needs these before it can claim user-controlled
disclosure across applications.

## Actions and deployment boundary

Capability grants and shell/file/search implementations exist, but they are
experimental rather than hardened isolation. Do not enable shell or file tools
for untrusted instructions. See [security limits](../SECURITY.md).

A learned interruption policy, automatic desktop routines, resumption cards and
wearable capture are not implemented. Existing proactive timing uses configured
logic; a future bandit or R-STDP action policy is a research proposal.

The former Tauri Pet-Face application was removed. Backend voice endpoints
remain; removal of the visual client did not remove microphone/transcription code.

## Evidence

Implementation is not the same as a validated product. See
[verification scope](VERIFICATION.md) for reproducible tests and known failures.
Proposed comparisons must use matched inputs, feedback budgets, held-out time
periods and simple baselines. Report failures and resource cost alongside accuracy.
