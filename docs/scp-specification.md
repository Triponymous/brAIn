# SCP — Spiking Communication Protocol

## Version 0.1.0 (Draft)

> Historical design draft, not the current external integration contract.
> The internal implementation now lives in `bridge/scp_schema.py`,
> `bridge/scp_server.py` and `bridge/scp_client.py`. Some paths include actions.
> For the separate query-only external adapter, use [MCP.md](MCP.md);
> for runtime boundaries, use [the architecture guide](living-desktop-manager.md).

### Abstract

SCP (Spiking Communication Protocol) is an open protocol for bidirectional communication between Spiking Neural Networks (SNNs) and Large Language Models (LLMs). It provides a standardized interface for LLMs to query neural state, receive events from the SNN, send feedback actions, and adopt SNN-driven personality modes.

SCP is to neural substrates what MCP is to external tools: a formal specification that enables any LLM to communicate with any SNN, regardless of implementation.

### Motivation

Current SNN+LLM systems inject neural state as unstructured text into LLM prompts. This approach:
- Is model-dependent (each LLM interprets prompts differently)
- Provides no feedback channel (LLM cannot affect SNN state)
- Mixes data with instructions (sensor values buried in personality text)
- Cannot be tested independently (prompt is a black box)

SCP solves this by separating concerns into typed messages with defined schemas.

---

## 1. Architecture

```
+------------------+         SCP Messages          +------------------+
|                  |  ---- Query/Response ------>   |                  |
|   Brain Server   |  <---- Events -----------      |   LLM Client     |
|   (SNN Runtime)  |  <---- Actions ----------      |   (Any LLM)      |
|                  |  ---- Personality -------->     |                  |
+------------------+                                +------------------+
        |                                                    |
   [SNN: Spikes,                                    [Text Generation,
    Modulators,                                      Tool Calling,
    Clusters,                                        User Interaction]
    STDP Weights]
```

### Roles

- **Brain Server**: Wraps an SNN runtime. Exposes neural state via Query responses, pushes Events on state changes, accepts Actions that affect SNN dynamics.
- **LLM Client**: Consumes Brain Server state. Can be any LLM (local or cloud). Uses Queries to read state, processes Events for context, sends Actions for feedback.
- **Adapter Layer** (optional): Translates SCP messages into model-specific prompt formats. Enables the same Brain Server to work with Qwen, Gemma, Claude, Llama, etc.

---

## 2. Message Types

All messages are JSON objects with a `type` field.

### 2.1 Query (Client -> Server)

The LLM requests specific neural state.

```json
{
  "type": "query",
  "method": "brain.<domain>",
  "id": "<unique-request-id>",
  "params": {}
}
```

**Defined Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `brain.emotion` | Emotional state + trend | Current emotion, confidence, 3-min trend |
| `brain.pattern` | Active cluster + sensors | What pattern is recognized, with what confidence |
| `brain.memory` | WM state + recent patterns | Working memory contents |
| `brain.session` | Session context | Duration, habits, break status |
| `brain.learning` | SNN training state | Neuron specialization, cluster count, age |
| `brain.full` | All of the above | Complete brain state snapshot |

### 2.2 Response (Server -> Client)

```json
{
  "type": "response",
  "id": "<matching-request-id>",
  "result": { ... }
}
```

**Response Schemas by Method:**

#### brain.emotion
```json
{
  "state": "content|curious|alert|focused|stressed|drowsy",
  "confidence": 0.85,
  "duration_seconds": 180,
  "trend": {
    "window_seconds": 180,
    "avg": {"DA": 0.006, "NE": 0.003, "ACh": 0.048, "5HT": 0.047},
    "peak": {"DA": 0.015, "NE": 0.012, "ACh": 0.052, "5HT": 0.049},
    "direction": "stable|rising|falling"
  },
  "reason": "Ruhige Session ohne Ueberraschungen"
}
```

#### brain.pattern
```json
{
  "cluster_id": 5,
  "label": null,
  "confidence": 0.51,
  "observation_count": 435,
  "sensors": {
    "app": "Claude",
    "keyboard": "still",
    "mouse": "gelegentlich",
    "mic": "still",
    "idle_seconds": 0
  },
  "suggested_label": "Browsing in Claude"
}
```

#### brain.memory
```json
{
  "wm_active": 5,
  "wm_capacity": 20,
  "recent_patterns": [
    {"cluster_id": 3, "label": "Coding", "ago_seconds": 120}
  ]
}
```

#### brain.session
```json
{
  "active_minutes": 47,
  "needs_break": false,
  "in_flow": false,
  "in_meeting": false,
  "habit_context": "Sonntag 23h: normalerweise idle",
  "anomalies": []
}
```

#### brain.learning
```json
{
  "specialized_neurons": 52,
  "total_neurons": 200,
  "specialization_pct": 26.0,
  "cluster_count": 5,
  "age_ticks": 10000000,
  "age_days": 1.2
}
```

### 2.3 Event (Server -> Client)

The SNN pushes state changes without being asked.

```json
{
  "type": "event",
  "method": "brain.<event_type>",
  "params": { ... }
}
```

**Defined Events:**

| Event | Trigger | Description |
|-------|---------|-------------|
| `brain.pattern_changed` | ConceptTracker transition | Pattern switched |
| `brain.emotion_changed` | Emotional state shift | Mood changed |
| `brain.stress_detected` | Sustained NE + low 5HT | User appears stressed |
| `brain.flow_detected` | Sustained same-app + typing | User in flow state |
| `brain.break_needed` | 90+ min without pause | Break reminder |
| `brain.new_pattern` | Unknown cluster active | Novel activity detected |
| `brain.label_needed` | Unlabeled cluster >1min | Pet should ask for label |

#### Example: brain.pattern_changed
```json
{
  "type": "event",
  "method": "brain.pattern_changed",
  "params": {
    "from": {"cluster_id": 3, "label": "Coding"},
    "to": {"cluster_id": 5, "label": null},
    "sensors": {"app": "Chrome", "keyboard": "still"}
  }
}
```

### 2.4 Action (Client -> Server)

The LLM sends feedback that affects SNN dynamics.

```json
{
  "type": "action",
  "method": "brain.<action_type>",
  "params": { ... }
}
```

**Defined Actions:**

| Action | Effect on SNN | Description |
|--------|--------------|-------------|
| `brain.reward` | DA += 0.02, 5HT += 0.01 | Positive signal (label confirmed) |
| `brain.correct` | NE += 0.01, DA += 0.005 | Error signal (label corrected) |
| `brain.label` | ConceptTracker.set_label() | Assign name to cluster |
| `brain.engage` | ACh += 0.005, DA += 0.002 | User is engaged (fast replies) |
| `brain.disengage` | 5HT -= 0.005 | User lost interest |

#### Example: brain.label
```json
{
  "type": "action",
  "method": "brain.label",
  "params": {
    "cluster_id": 5,
    "label": "Browsing in Chrome",
    "source": "user_confirmed"
  }
}
```

### 2.5 Personality (Server -> Client)

The SNN dictates how the LLM should behave. This is NOT a prompt template — it's a declarative specification that the Adapter Layer translates into model-specific instructions.

```json
{
  "type": "personality",
  "params": {
    "mode": "content",
    "style": {
      "tone": "warm, reflektiv",
      "length": "2-4 Saetze",
      "questions": false,
      "urgency": "low"
    },
    "constraints": [
      "keine Emojis",
      "keine Hilfe anbieten",
      "keine erfundenen Faehigkeiten",
      "nicht den Sinnen widersprechen"
    ],
    "priority_topic": null
  }
}
```

---

## 3. Transport

SCP is transport-agnostic. Reference implementations:

- **In-process** (Python function calls): For single-machine SNN+LLM systems
- **WebSocket**: For distributed setups (SNN on device, LLM in cloud)
- **JSON-RPC over HTTP**: For REST-compatible systems

The reference implementation uses in-process calls (Python method dispatch).

---

## 4. Adapter Layer

The Adapter Layer translates SCP messages into model-specific prompt formats. This is the ONLY model-dependent component.

```
SCP Messages (model-independent JSON)
         |
    [Adapter Layer]
         |
    Model-specific prompt string
```

Adapters are registered per model family:
- `QwenAdapter`: direct instructions, structured headers
- `GemmaAdapter`: example-based, softer tone
- `ClaudeAdapter`: XML tags, constraint blocks
- `GenericAdapter`: minimal, works with any model

---

## 5. Comparison with MCP

| Aspect | MCP | SCP |
|--------|-----|-----|
| **Direction** | LLM -> External World | LLM <-> Neural Substrate |
| **Data** | Files, APIs, databases | Spikes, modulators, weights |
| **Events** | Tool results | Neural state changes |
| **Feedback** | N/A | Modulator injection |
| **Personality** | N/A | SNN-driven behavior modes |
| **Transport** | stdio, SSE | In-process, WebSocket |

---

## 6. Reference Implementation

The reference implementation is part of the brAIn project:
- `bridge/scp_server.py` — Brain Server
- `bridge/scp_client.py` — LLM Client
- `bridge/scp_schema.py` — JSON Schema definitions
- `bridge/model_adapter.py` — Adapter Layer
- `bridge/feedback.py` — Action handlers
