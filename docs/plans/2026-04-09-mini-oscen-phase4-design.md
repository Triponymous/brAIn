# Mini-OSCEN Phase 4 — Capability Wishlist + Grant System Design

**Date:** 2026-04-09
**Status:** Approved, ready for implementation planning

## Vision

The pet starts with a minimal toolkit (memory tools + TTS). It observes the user's desktop life, forms stable concept neurons, and when a labeled concept matches a tool in the capability catalog, the LLM bridge proposes: "I notice you do X often — should I learn to help with that?" The user grants → tool is registered at runtime → persisted across restarts. Each pet's capability set grows uniquely from its user's actual behavior.

## Architecture

Four components: Capability Catalog (code), Wish Detector (background coroutine), Grant Flow (chat + dashboard UI), Runtime Tool Registry (extends MemoryTools).

## Capability Catalog

Python dict with three entries for Phase 4:

1. **web_search** — DuckDuckGo HTML API (no API key), returns top-5 results. Trigger keywords: google, suchen, recherche, search, nachschlagen.
2. **shell** — asyncio subprocess with allowlist (ls, cat, wc, date, which, brew list, pip list) + blocklist (rm, sudo, mv, chmod, kill). Trigger keywords: terminal, command, befehl, shell, cli.
3. **local_files** — read/write text files in a configurable directory (default ~/Notes/). Trigger keywords: notizen, notes, datei, file, markdown, schreiben.

Each entry: name, description, trigger_keywords, tool_schema, execute_fn, default_enabled=False.

## Wish Detector

Background coroutine running every 5 minutes:
1. Read top-20 concepts by 24h activation
2. Filter to labeled concepts only
3. Match labels against catalog trigger keywords (substring match)
4. Generate wish if: match score > threshold, concept active ≥10× in 24h, tool not already granted, not denied in last 7 days
5. Store wish in SQLite
6. Trigger LLM to formulate natural-language proposal
7. Show in dashboard as special grant-request message

## Grant Flow

User sees a chat message with [Grant] [Deny] [Later] buttons.
- Grant: wish → granted, tool registered, persisted to SQLite
- Deny: wish → denied, 7-day cooldown
- Later: wish stays pending, re-proposed in 24h

## Runtime Tool Registry

Wraps existing MemoryTools. tool_definitions() returns memory tools + all granted capability tools. execute() dispatches to either. On daemon start, loads grants from SQLite.

## SQLite Schema Extension

```sql
capability_wishes(id PK, tool_name, trigger_concept_id, trigger_concept_label, reason, status, created_at, resolved_at)
capability_grants(tool_name PK, granted_at, trigger_wish_id FK, params JSON)
```

## Non-Goals

- No automatic grant (user must confirm)
- No tool revoke UI (SQLite manual delete + restart)
- No tool parameter config UI
- No conversation-history matching (wishes from concept labels only)
- No proactive tool use (only when user asks in chat; proactive = Phase 5)

## Success Criteria

1. Catalog has 3 tools, all disabled by default
2. Wish detector identifies stable labeled concepts and proposes matching tools
3. Grant flow works end-to-end: wish → user approval → tool available in next chat
4. Granted tools survive daemon restart (SQLite persistence)
5. web_search, shell, local_files all work when granted and called by the LLM
6. Test count grows from 134 to ~150
