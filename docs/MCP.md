# MCP context interface

Implemented in [server/mcp.py](../server/mcp.py) and
[server/tools.py](../server/tools.py). Reviewed **2026-09-15**.

## Transport and runtime

`brain-mcp` is a **stdio** server launched by a compatible client.
It proxies queries to the persistent daemon's `POST /api/tools/{name}`.
The default daemon URL is `http://127.0.0.1:8000`; `BRAIN_URL` or `--url`
can override it. Use trusted loopback endpoints only.

```sh
.venv/bin/brain-mcp --help

# Executable/arguments to configure in a client supporting local stdio:
.venv/bin/brain-mcp --url http://127.0.0.1:8000
```

Use an absolute executable path when configuring another application.
Do not type into its stdio transport manually or treat it as an HTTP server.
Listing tools works without a running daemon; calls report the unavailable daemon.

The observer on port 8001 is independent and does not expose these endpoints.
Starting MCP does not start a model, create learned history or enable capture.

## Nine tools

| Tool | Query |
| --- | --- |
| `brain_state` | Current model state, labels, activity metadata and modulators; which sources are shared and whether the model is paused |
| `brain_history` | Time-bucketed episode summaries |
| `brain_concept` | A concept profile |
| `brain_felt` | Learned label prototypes |
| `brain_habits` | Derived routine summaries |
| `brain_anomalies` | Deviations from historical summaries |
| `brain_why` | Model-mechanism explanation |
| `brain_recall` | Recorded pattern intervals |
| `brain_experience` | Interaction-event summary |

Exact argument schemas come from `BrainTools.tool_definitions()`, not a second
hand-maintained schema. These tools are observational interfaces to a research
model: `brain_why` is not a causal explanation of a person's feelings.

## Permissions and disclosure

No model mutation, labeling, reward or OS/file action is exposed through this
adapter. **Query calls are nevertheless logged** with tool arguments and the
`mcp` source tag in the daemon's experience database.

Results may contain personal labels, app names, activity patterns and numbers.
A sense the user does not share is reported as `null`, never as zero.
A host can forward them to its model provider. The internal daemon cloud setting
does not govern an external client. Review the host's consent/storage behaviour.

Per-client field scopes, expiry, pairing, revocation and authenticated remote MCP
are not implemented. Do not publish the daemon or proxy behind a public tunnel.
[SCP](scp-specification.md) is a separate internal protocol with action paths.

## Client compatibility and planned skill

A model and a host application are different things: the host must support
launching a local stdio server. This repository does not establish universal
support for every ChatGPT or Claude surface. Check the selected host's current
configuration instructions; there is no bundled cross-client installer.

A reusable skill explaining when to query, how to handle stale/unknown state and
what may be disclosed is planned. It must not tell a model to equate inferred
labels with medical facts or to call tools without the host's approval.

Context/tool access does not fine-tune model weights or guarantee long-term
memory in the receiving application. [Roadmap](ROADMAP.md).
