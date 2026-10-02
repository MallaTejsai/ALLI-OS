"""Phase 2 — Canonical SQLite schema (rules 50/51/75). New DB, new version identity."""
from __future__ import annotations
SCHEMA_VERSION = "v0.6.0-new"
DB_FILE = "alli_v0_6_new.db"
# Canonical tables (rule 7): memory_claims, entities, relationships,
# episodes, evidence, provenance, lifecycle_history, temporal_scopes.
