# Research roadmap

Reviewed **2026-09-15**. This is a proposal, not a release schedule.
[Current implementation](../README.md) · [Architecture](living-desktop-manager.md)

## 1. Establish trustworthy sessions

**Current:** opt-in ephemeral observation, persistent daemon logs, visible
capture controls in Observatory, explicit observed-window export. The persistent
daemon has per-source consent: every source starts off, the choice survives
restarts without widening, and switching a source off stops its acquisition.

**Next:** consent-bound recording, stable session identifiers, timestamps,
versioned labels, restart behaviour, retention and deletion across derived data.

**Acceptance:** disabling a source stops acquisition; restarting never silently
widens consent; exported provenance matches the model; deletion covers documented
stores and backups. Tests must exercise transitions, not only static flags.

## 2. Test whether personal learning helps

Compare simple rules and a classical model against brAIn using identical inputs,
feedback opportunities and held-out days. Keep overlapping time windows out of
both training and test sets. Use an explicit user report for the target rather
than treating model chemistry as ground truth.

Measure errors, abstentions, calibration, drift, user effort, CPU/time cost and
useful outcomes. Publish synthetic fixtures and protocols; never personal traces
by default. A null result is a valid result.

An “Open NeuroLab” comparison interface and per-source ablations are candidates
for making these experiments inspectable. Neither is currently a shipped platform.

## 3. Make personal context portable and controllable

**Current:** nine daemon query tools, HTTP adapter and a stdio MCP server.

**Next:** per-client and per-field consent, freshness/expiry, provenance,
revocation, redacted responses and a reusable client guide/skill.
MCP support depends on the host; an arbitrary chat window cannot be assumed to
launch a local stdio process. No model-weight training is implied.

**Acceptance:** a client cannot retrieve unapproved fields; expired context is
explicitly stale; revocation blocks subsequent access; unavailable observations
are not replaced with synthetic values. Document exactly what a cloud host sees.

## 4. Evaluate behaviour before automating it

A contextual bandit for asking, staying silent or offering help is a proposal.
Experience logs provide observations, not a validated reward function.
A rise in NE or a held 5HT value is not proof that an interruption was harmful or helpful.

Start with opt-in suggestions and user feedback. Compare against fixed timing and
a no-suggestion condition. Track declined suggestions and interruption burden.
Do not let a learned policy execute arbitrary shell commands.

A manual **resumption card** (“next step” plus explicitly chosen links) is a
small product experiment. Compare with an ordinary note before adding automatic
selection. Coarse desktop metadata does not reveal a document's contents or intent.

## Optional sensors and circuit experiments

| Idea | Current status | Gate before integration |
| --- | --- | --- |
| Mel/RMS + local VAD | Offline pipeline | Legally usable recordings, independent source groups, speech/music/singing/podcast counterexamples, measured incremental benefit |
| Wearables: heart rate, HRV, temperature | Not connected | Explicit consent, source-specific metrics, measurement age, secure device handoff, benefit beyond desktop-only baseline |
| Fly-inspired sparse inhibition | Proposal | Matched random/top-k controls, same seeds and exposure, held-out performance and resource cost |
| Actual model connectivity in the atlas | Not in live stream | Export real weights/edges with provenance; distinguish computational roles from anatomical labels |
| Learned desktop routines | Proposal | Narrow action allowlist, hardened execution boundary, explicit grants and reversible actions |
| LLM fine-tuning | Deferred | Consented dataset and evidence that simpler context/tool approaches are insufficient |
| Hardware companion | Proposal | A concrete maintained implementation and measured power budget |

Do not add VAD, a new topology and wearable features simultaneously: separate
ablations are needed to attribute any improvement. Screen capture is not part of
the current observation scope.

## Non-goals for the next iteration

No medical/emotion diagnosis, consciousness claim, commercial acquisition promise,
universal smartwatch connector or autonomous access to a user's computer.
No claim of novelty or superiority without literature review and empirical evidence.
