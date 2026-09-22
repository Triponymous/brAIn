# Security status

brAIn is a **trusted-local research prototype**, not an audited desktop security
product. No supported public or multi-user deployment is offered.

## Current boundaries

- Keep observer, daemon and control endpoints on loopback.
- The daemon and control server refuse requests from other web origins and
  from non-loopback `Host` names; the daemon's browser clients are the
  training console on port 8900 and any origin listed under
  `daemon.allowed_origins` in `config.json`. Before this check they allowed every origin,
  so any website open in the same browser could grant tools or stop the daemon.
  Local processes are still not authenticated.
- Do not expose them via a public tunnel, wildcard bind or reverse proxy.
- Observatory capture controls govern only the observer, not the broader daemon.
- Model weights, activity metadata and user labels may be sensitive.
- MCP query results may leave the machine through the chosen host.
- The experimental capability system is **not a hardened sandbox**. Do not grant
  shell or file capabilities for untrusted model instructions. Grants express
  permission but do not establish process/filesystem isolation.

Before broader deployment, action execution, path confinement, authentication,
origin enforcement, data retention and consent need dedicated security review.
The default quick start deliberately uses the narrower observer.

## Dependencies and files

Respect the constraints in `pyproject.toml`; do not silently lower a version
floor. Test compatibility in an isolated environment before replacing a running
research environment. Declared minimums alone are not proof of security.

Load checkpoints, ONNX files and other serialized assets only from trusted sources.
Preserve upstream licenses and verify expected fingerprints where provided.
Do not put credentials in tracked config or personal data in issues/test fixtures.

## Reporting

If GitHub offers **Security → Report a vulnerability** for this repository, use
that private channel. Otherwise contact the maintainer privately through the
contact information on their GitHub profile before sharing exploit details.
Private reporting availability is not guaranteed by this file.

For ordinary non-sensitive bugs, use GitHub Issues with synthetic reproduction
steps. Do not attach private checkpoints, recordings, API keys or activity logs.

No security fix, independent audit or certification is implied by this notice.
