# brAIn: Living Desktop Manager — direction and architecture

snnTorch, sharing this project on LinkedIn and their blog, called it
*brAIn: Living Desktop Manager*. This document is the engineering side of
that name: what the thing is meant to become, what is already there, what is
missing, and in which order it gets built. The README's *Thesis* section is
the short version.

## The goal, in the owner's words

An LLM that gets to know me through a spiking network — through emotions and
actions rather than text — and acts for me. An LLM that learns the way an
animal or a human does.

## The decision: where the life is

**The animal is the spiking network. The LLM is its voice and its hands.**

An LLM's weights move by gradient descent over text, in batches, forgetting
as they go; that is not how an animal learns, and no amount of fine-tuning
on a laptop makes it so. The spiking network in `brain/` already learns the
animal way: synapses that follow what happens (STDP, BCM), feelings that are
the shape of prediction error (the emergent modulator driver), attachment to
one human, memory that *is* the network. So the LLM never becomes the
animal. It becomes a faithful, fluent extension of one:

1. **It understands the animal completely — as tools, not as a paragraph.**
   Every answer is conditioned on a state grown by living with this person.
2. **The animal teaches it how to behave.** When to be silent, when to
   speak, what to offer — a policy learned from consequence, on device.

Both stand on the **experience log**. Nothing in this plan changes an LLM
weight. The door to fine-tuning stays open and is opened, if ever, by the
numbers the log produces.

## What exists (September 2026)

| Layer | Where | State |
|---|---|---|
| Organism: LIF, expansion, WTA concept layer, WM, STDP/BCM, emergent DA/NE/ACh/5HT | `brain/` | done, benchmarked |
| Learned felt-state vocabulary + active-learning asks | `bridge/felt_state.py`, `bridge/proactive.py`, `server/feel.py` | done |
| Stable concept clusters with sensor profiles and label suggestions | `brain/concept_tracker.py`, `bridge/exporter.py` | done |
| Episodic history (one snapshot per ~10 s), habits, anomalies | `bridge/episode_log.py`, `habit_miner.py`, `anomaly.py` | done |
| SCP: typed queries, events, actions between brain and LLM | `bridge/scp_*.py`, `docs/scp-specification.md` | done (v2), used as prompt material |
| Narrator: brain state as prose for the prompt | `bridge/snn_narrator.py`, `emotional_prompt.py` | done — the *thin* interface this plan replaces |
| Capabilities: wish detection, grants, tools (search, shell, files) | `capabilities/` | skeleton, gated |
| Always-on: launchd → control server → supervised daemon | `server/control.py`, `scripts/` | done |
| **Experience log** | `bridge/experience.py`, `server/experience.py` | **done — this step** |
| Brain as tools for the LLM | — | next |
| Learned behaviour policy | — | after that |

## The experience log

One row per event: who acted, what they did, the organism's state at that
moment, and — written ~120 s later — what followed.

```
experiences(id, ts, tick,
            actor      'pet' | 'human' | 'llm',
            kind       ask_label | notify | teach_felt | dismiss_ask | chat |
                       label_concept | brain.reward | brain.correct | brain.label | ...
            payload    JSON, never content,
            felt_label, felt_conf, cluster, sleep, signature[6],   -- state at ts
            after_signature[6], after_ts,                          -- consequence
            response_kind 'answered' | 'dismissed', response_id)   -- how the human replied
```

- **Recorded by** the proactive engine (asks, notifications), `/api/feel`
  (teach, dismiss), `/api/chat` (that a conversation happened), `/api/label`,
  and every SCP action the LLM takes on the brain.
- **Consequence** is the 180 s modulator signature 120 s after the event —
  the same six numbers the felt-state model reads. If the daemon was down in
  between, the row is closed without a consequence rather than given today's
  mood.
- **Facts, not judgements.** The log never stores a reward; whoever reads it
  computes one, so the reward function can change without losing history.
- **Never content.** No chat text, no keystrokes, no audio. Labels the human
  chose, categories, app names, counts. The privacy line of the README holds.
- **Read** via `GET /api/experience?limit=&hours=`: recent rows plus a
  summary — counts per actor and kind, ask/answer/dismiss rates, and the mean
  signature change after each kind of pet action. After one week of running,
  that summary is the first real answer to *is it learning me?*

## Next: the brain as tools

Replace the narrated paragraph with tool calls the LLM makes while it
reasons (local Qwen via Ollama and the cloud model both support tool use;
the chat endpoint already executes tool calls through `ToolRegistry`).

| Tool | Returns | Backed by (exists) |
|---|---|---|
| `brain.state()` | felt-state + confidence, modulators, prediction-error internals, current cluster, WM occupancy, sleep | `felt_state`, `modulators`, `core.py` driver, `concept_tracker` |
| `brain.history(since, until, step)` | clusters and modulators over time | `episode_log` |
| `brain.concept(id)` | label, sensor profile, first seen, times seen, stability | `exporter.get_concept_profile`, `concept_tracker` |
| `brain.felt(label)` | prototype, count, when it occurs, what precedes/follows | `felt_state`, experience log |
| `brain.habits()` / `brain.anomalies()` | hourly and weekly profile, deviations | `habit_miner`, `anomaly` |
| `brain.why(modulator)` | what drove it | `synapse_explainer` |
| `brain.recall(question)` | episodic search over history | `episode_log` + experience log |
| `brain.experience(hours)` | what the pet did and how it went | experience log |

SCP already defines the query/action/event envelope; the tool schema is a
projection of it. The narrator stays as a fallback for models without tool
use.

## Then: the learned behaviour policy

- **Actions:** stay silent · notify · ask a question · suggest a break ·
  *offer* a task.
- **Context:** felt-state label, cluster, hour bucket, modulator band, sleep.
- **Reward, computed from the log:** the human's response (answered +,
  dismissed −, silence 0) plus the affect consequence (ΔNE up is bad,
  Δ5HT held is good). The organism's own feeling after acting is the
  teacher — that is the animal part.
- **Learner:** a contextual bandit (Thompson sampling per context/action),
  persisted in the checkpoint, fully inspectable. It replaces the hand-tuned
  intervals and priority order in `proactive.py` — the same move the modulator
  driver made for emotion: emergent instead of scripted.
- **Safety:** the policy decides whether to *offer*; nothing executes without
  a grant (`capabilities/grants.py`). A bandit never runs a shell command.

## Later, by the numbers: fine-tuning

With hundreds of logged experiences and an explicit, opt-in transcript store
(the log itself never holds text), a LoRA adapter on the local model becomes
an experiment with a dataset rather than a hope. Judge it then.

## Order of work

1. Live always-on — done.
2. Experience log — done. Let it run; read `/api/experience` after a week.
3. Brain as tools — wire the table above into `ToolRegistry`; the dashboard's
   Language view shows the calls.
4. Learned policy — bandit over the log; the dashboard's Growth view shows
   what it learned.
5. Fine-tuning — decide with data.

## Non-goals

Training LLM weights from SNN state directly; a general assistant; anything
that stores content. The point is one individual that knows one person.
