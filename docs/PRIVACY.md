# Privacy and data lifecycle

Reviewed against source on **2026-09-15**. This is an implementation inventory,
not a legal certification or a promise that derived data is anonymous.

## Runtime boundaries

| Path | Inputs and outputs | Retention / control |
| --- | --- | --- |
| Static Observatory demo | Synthetic fixtures; user-entered demo annotations | Demo annotations stay in the tab and explicit exports; tour-seen flag and language are stored in the browser |
| Observer (`server.observe`) | Four opt-in coarse desktop metadata channels; model output | 512-frame server ring, up to 200 per response/UI window; no checkpoint; export is explicit |
| Persistent daemon (`server.braind`) | Five sources, each off until shared: key and pointer event rates, idle time, app names (front and open apps), microphone features; taught labels | Per-source consent under **Live session → Persistent brain**, saved in `consent.json`; checkpoint and companion SQLite databases; the session runner's switches do not govern it. Live frames for the dashboard stay in a 512-frame RAM window and are sampled only while a view reads them |
| Daemon chat | Submitted text, supplied history and model/app context | Sent to the selected LLM backend; the experience event records a character count, not the submitted chat body |
| Voice endpoints | Explicitly requested microphone recording/transcription and synthesized output | Refused unless the microphone source is shared, and always in mock mode; at most 30 s per request; discarded if the microphone is switched off while recording. Processes actual speech; therefore a repository-wide “no transcripts/content” claim would be false |
| External MCP | State, labels, app information, patterns and query results | Host can store/disclose returned data; daemon logs query name, arguments and source |
| Optional action tools | Granted search, shell and file operations | Experimental trust boundary; see [SECURITY.md](../SECURITY.md) |

No observer source captures key text, pointer coordinates, window titles,
background-app lists, raw audio or screenshots. The observer briefly reads the
foreground app name to categorize it; it does not retain/export that name.
Those restrictions must not be generalized to the broader daemon. The live
frames the dashboard reads from the daemon carry the app source as a broad
category and the microphone as loudness only; that view does not narrow what
the daemon itself uses or stores.

## Persistent files

By default, the daemon uses the `checkpoints/` directory:

- `braind.sqlite`: model state, learned labels and associated persistent state.
- `episodes.db`: activity summaries, including app/label information; default
  episode pruning is 90 days.
- `experience.db`: event metadata, model signatures, responses and tool arguments;
  events older than 90 days are pruned when the daemon starts and once a day.
- `grants.sqlite`: capability permissions.
- `consent.json`: which sources are shared, when each choice changed, and a
  revision number. The daemon reads it on start; if it is missing or
  unreadable, nothing is shared. Mock mode uses `consent-mock.json` instead.
- Checkpoint backups and runtime logs may also remain.

Removing the main checkpoint alone is not complete erasure. `python -m
server.braind erase` lists every file above with its SQLite side files, the
backups, the consent choice and, for the default directory, the pidfile and the
login service's logs; `--yes` deletes them. It refuses while a daemon uses
those files: every daemon holds a lock on its checkpoint, whatever its port.
Explicit exports, copies an MCP host or LLM provider received, and OS-level
backups are outside its reach. The dashboard sets per-source consent; there is
no deletion button in a UI yet.

## Stop is not erase

- Observer **Stop all capture** stops its source acquisition and further steps.
- Daemon sources (**Live session → Persistent brain**): switching a source off
  stops its listener, audio stream or polling and drops its last value; with
  nothing shared the model does not step. A stop holds for the running daemon
  even if saving the choice fails, and the dashboard reports that failure.
- Erasing is a separate, explicit step: `python -m server.braind erase --yes`
  with the daemon stopped.
- **Disconnect view**, freezing and closing the tab do not stop collection.
- Stopping a source does not erase historical frames, learned weights or exports.
- Observer process termination releases its model/ring buffer; this is not a
  secure-memory-erasure guarantee.
- Stop the persistent daemon separately with **Stop daemon** in the dashboard
  or via its process lifecycle.
  Its checkpoint and logs intentionally survive shutdown.
- OS permissions are distinct from application capture switches.

## Local does not mean inaccessible

Origin/Host checks in the observer, the daemon and the control server protect
a limited browser boundary: other websites and DNS-rebinding pages are refused,
and the daemon accepts browser requests only from the dashboard (127.0.0.1 or
localhost on port 8900, where the control server serves it, or 4178, the static
preview) and origins listed under `daemon.allowed_origins` in `config.json`; the
control server accepts the dashboard only. Other authorized local processes are
outside that protection.
The broader daemon/control APIs are trusted-local research interfaces, not
authenticated multi-user services.

The external MCP adapter does not offer label/teach/reward/shell/file actions,
but query calls are recorded as experiences. Arguments may contain user-provided
labels or search strings. “Read-only” must not be interpreted as no persistence,
no personal information, or no observable side effects.

A cloud-hosted assistant may send tool results off the machine even when brAIn's
own `cloud_enabled` setting is false. That flag governs the internal router only.
No per-client field-level disclosure controls are implemented yet.

## Before adding data sources

Wearables and live VAD require separate consent, source status, timestamps,
capture-level stop and a measured incremental benefit.
No WHOOP/HealthKit account or wearable connection is implemented.
Do not treat vendor stress/recovery scores as ground truth for the same prediction.

Use synthetic fixtures in public issues and screenshots. Never assume model
weights, inferred labels or aggregate activity patterns are anonymous.
Do not commit personal data just because Git can store it.
