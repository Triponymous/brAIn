# Documentation

Current entry points, reviewed **2026-09-15**. Public overview documents are in
English; detailed design records may be in German.

Some historical shell examples use the optional `rtk proxy` output wrapper.
It is not a brAIn dependency; run the command after that prefix directly if
you do not have it. Current setup commands do not require it.

## Start here

| Question | Document |
| --- | --- |
| What is implemented? | [Project overview](../README.md) |
| How do I run it safely? | [Runtime guide](RUNNING.md) |
| What is collected, saved or disclosed? | [Privacy and data lifecycle](PRIVACY.md) |
| How are the two runtimes connected? | [Architecture](living-desktop-manager.md) — they are independent |
| How can another AI client query brAIn? | [MCP interface](MCP.md) |
| Which ideas are next? | [Roadmap](ROADMAP.md) |
| What was actually tested? | [Verification](VERIFICATION.md) |
| How do I contribute? | [Contributor guide](../CONTRIBUTING.md) |
| What is not safe to expose? | [Security](../SECURITY.md) |

## Observatory implementation records

- [Current live data and capture controls](dashboard-concepts/LIVE-DATA.md)
- [English/German introduction](dashboard-concepts/EINFUEHRUNG.md)
- [Research workspace: synthetic demonstration pages](dashboard-concepts/RESEARCH-WORKSPACE.md)
- [3D implementation](dashboard-concepts/BRAIN-3D.md)
- [Current anatomy and role-color mapping](../3d/ANATOMY-COLOR.md)
- [Asset attribution](dashboard-concepts/assets/brain-LICENSE.md)
- [Asset inventory](dashboard-concepts/assets/README.md)

The [original concepts](dashboard-concepts/KONZEPT.md),
[early Observatory design](dashboard-concepts/OBSERVATORY.md) and
[2D visual study](dashboard-concepts/BRAIN-VISUAL.md) are historical design records.
Their dated tests are not current release certification.
The [SCP draft](scp-specification.md) is a historical protocol proposal; current
behaviour is defined by the implementation and tests, not every example in it.

## Research proposals

- [Offline audio/VAD experiment](experiments/audio-vad.md): pipeline implemented;
  a real-recording improvement has not been established.
- [Connectome-inspired experiments](experiments/connectome-inspiration.md):
  proposed ablations, not an imported fly brain or current learning rule.
- [Wearables and study gates](ROADMAP.md#optional-sensors-and-circuit-experiments):
  no WHOOP, HealthKit or smartwatch connector implemented.

Historical external-source summaries retain their original dates. Recheck vendor
APIs, licenses and platform requirements before implementing an integration.
