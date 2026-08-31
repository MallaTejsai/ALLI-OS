"""Safe memory manager: validates ADD/UPDATE/DELETE; never trusts model file writes."""

from __future__ import annotations

import re
from typing import Optional

from . import config
from .embeddings import cosine_similarity
from .extractor import (
    MemoryExtractor,
    goal_match_score,
    has_explicit_change_signal,
    looks_like_person_name,
    normalize_goal_text,
    normalize_topic_subject,
    normalize_topic_text,
    preference_polarity,
    preference_subject,
    preference_subjects_match,
    normalize_preference_subject,
    strip_goal_filler,
    topic_match_score,
)
from .store import (
    MemoryStore,
    build_embedding_text,
    default_importance_for,
    infer_fact_type,
)


def normalized_content(content: str) -> str:
    words = re.findall(r"[a-zA-Z0-9+#]+", str(content).lower())
    ignored = {"a", "an", "the", "user"}
    return " ".join(word for word in words if word not in ignored)


class SafeMemoryManager:
    """Apply extracted operations with deterministic validation rules."""

    def __init__(
        self,
        store: MemoryStore,
        extractor: Optional[MemoryExtractor] = None,
        *,
        duplicate_threshold: Optional[float] = None,
    ):
        self.store = store
        self.extractor = extractor or MemoryExtractor()
        self.duplicate_threshold = (
            duplicate_threshold
            if duplicate_threshold is not None
            else config.DUPLICATE_SIMILARITY_THRESHOLD
        )

    @property
    def memories(self) -> list[dict]:
        return self.store.memories

    @property
    def active_memories(self) -> list[dict]:
        return self.store.active_memories()

    def process_user_message(self, user_message: str) -> list[dict]:
        """
        Extract and apply memory operations.

        Returns a list of result dicts describing what happened, for logging/tests.
        """
        operations = self.extractor.extract(user_message)
        results = []
        for operation in operations:
            result = self.apply_operation(operation, user_message)
            if result:
                results.append(result)
        return results

    def apply_operation(
        self,
        operation: dict,
        user_message: str = "",
    ) -> Optional[dict]:
        action = str(operation.get("action", "")).upper()

        if action in {"UPSERT", "ADD"}:
            return self._apply_upsert_or_add(operation, user_message)

        if action == "REPLACE_GOAL":
            return self._apply_goal_replacement(operation)

        if action == "DELETE_GOAL":
            return self._apply_goal_delete(operation)

        if action == "DELETE_MATCH":
            return self._apply_delete_match(operation, user_message)

        if action in {"UPDATE", "DELETE"}:
            # Conservative path used if a structured op provides an ID.
            if not has_explicit_change_signal(user_message):
                return {
                    "status": "skipped",
                    "reason": "no_explicit_change_signal",
                }
            memory_id = operation.get("memory_id")
            if not isinstance(memory_id, str):
                return {"status": "skipped", "reason": "invalid_id"}
            if action == "DELETE":
                ok = self.store.delete_memory(memory_id)
                return {
                    "status": "deleted" if ok else "skipped",
                    "memory_id": memory_id,
                }
            updated = self.store.update_memory(
                memory_id,
                content=operation.get("content"),
                category=operation.get("category"),
                fact_type=operation.get("fact_type"),
                importance=operation.get("importance"),
            )
            return {
                "status": "updated" if updated else "skipped",
                "memory_id": memory_id,
                "memory": updated,
            }

        return {"status": "skipped", "reason": f"unknown_action:{action}"}

    def _apply_upsert_or_add(
        self,
        operation: dict,
        user_message: str,
    ) -> dict:
        category = operation.get("category", "other")
        content = str(operation.get("content", "")).strip()
        if not content or category not in config.VALID_CATEGORIES:
            return {"status": "skipped", "reason": "invalid_fields"}

        fact_type = operation.get("fact_type") or infer_fact_type(
            category, content
        )
        importance = operation.get("importance") or default_importance_for(
            category, fact_type
        )

        # Identity unique slots: name / nickname — never cross fact_types
        if (
            category == "identity"
            and fact_type in config.UNIQUE_IDENTITY_FACT_TYPES
        ):
            return self._upsert_unique_identity(
                content=content,
                fact_type=fact_type,
                importance=importance,
                value=operation.get("value"),
                user_message=user_message,
                explicit_change=bool(operation.get("explicit_change")),
            )

        # Preferences: same subject → update (including polarity reversal)
        if category == "preference":
            return self._upsert_preference(
                content=content,
                importance=importance,
                operation=operation,
            )

        # Skills / interests / learning: same subject may update; else ADD.
        # Never conflict across categories.
        if category in {"skill", "interest", "learning"}:
            return self._upsert_by_subject(
                category=category,
                content=content,
                fact_type=fact_type,
                importance=importance,
                operation=operation,
                user_message=user_message,
            )

        # Goals / projects / others: semantic + exact dedupe; otherwise ADD
        duplicate = self.find_duplicate(
            category=category,
            fact_type=fact_type,
            content=content,
        )
        if duplicate is not None:
            # Exact / semantic same fact → no new memory
            if normalized_content(duplicate.get("content", "")) == normalized_content(
                content
            ) or self._semantic_same_fact(duplicate, content):
                return {
                    "status": "unchanged",
                    "memory_id": duplicate["id"],
                    "memory": duplicate,
                }

        memory = self.store.add_memory(
            category=category,
            content=content,
            fact_type=fact_type,
            importance=importance,
            value=operation.get("value"),
            subject=operation.get("subject") or operation.get("preference_subject"),
        )
        return {"status": "added", "memory_id": memory["id"], "memory": memory}

    def _upsert_unique_identity(
        self,
        *,
        content: str,
        fact_type: str,
        importance: str,
        value: Optional[str] = None,
        user_message: str = "",
        explicit_change: bool = False,
    ) -> dict:
        # Refuse non-name payloads so interests never overwrite identity.
        resolved_value = (value or "").strip()
        if not resolved_value and " is " in content:
            resolved_value = content.split(" is ", 1)[-1].strip().rstrip(".")
        if fact_type == "name" and not looks_like_person_name(resolved_value):
            return {
                "status": "skipped",
                "reason": "identity_value_not_person_name",
            }

        existing = None
        for memory in self.active_memories:
            if (
                memory.get("category") == "identity"
                and memory.get("fact_type") == fact_type
            ):
                existing = memory
                break

        if existing is None:
            memory = self.store.add_memory(
                category="identity",
                content=content,
                fact_type=fact_type,
                importance=importance,
                value=resolved_value or value,
                subject=fact_type,
            )
            return {"status": "added", "memory_id": memory["id"], "memory": memory}

        if normalized_content(existing.get("content", "")) == normalized_content(
            content
        ):
            return {
                "status": "unchanged",
                "memory_id": existing["id"],
                "memory": existing,
            }

        # Name/nickname updates require an explicit identity statement or
        # change signal — never silent overwrite from unrelated text.
        lower_msg = (user_message or "").lower()
        allowed_update = (
            explicit_change
            or has_explicit_change_signal(user_message)
            or (
                fact_type == "name"
                and (
                    "my name" in lower_msg
                    or re.search(r"\b(?:i am|i'm)\s+" + re.escape(resolved_value), lower_msg, re.I)
                )
            )
            or (
                fact_type == "nickname"
                and ("nickname" in lower_msg or "nick name" in lower_msg)
            )
        )
        if not allowed_update:
            return {
                "status": "skipped",
                "reason": "no_explicit_identity_change",
                "memory_id": existing["id"],
            }

        updated = self.store.update_memory(
            existing["id"],
            content=content,
            fact_type=fact_type,
            importance=importance,
            value=resolved_value or value,
            subject=fact_type,
            archive_history=True,
        )
        return {
            "status": "updated",
            "memory_id": existing["id"],
            "memory": updated,
            "conflict_resolved": True,
        }

    def _upsert_by_subject(
        self,
        *,
        category: str,
        content: str,
        fact_type: str,
        importance: str,
        operation: dict,
        user_message: str = "",
    ) -> dict:
        subject = normalize_topic_subject(
            operation.get("subject")
            or operation.get("value")
            or content
        )
        if not subject:
            memory = self.store.add_memory(
                category=category,
                content=content,
                fact_type=fact_type,
                importance=importance,
                value=operation.get("value"),
                subject=subject,
            )
            return {"status": "added", "memory_id": memory["id"], "memory": memory}

        for memory in self.active_memories:
            # Strict: never match across categories.
            if memory.get("category") != category:
                continue
            old_subject = normalize_topic_subject(
                memory.get("subject") or memory.get("value") or memory.get("content", "")
            )
            if not old_subject or old_subject != subject:
                continue

            if normalized_content(memory.get("content", "")) == normalized_content(
                content
            ):
                return {
                    "status": "unchanged",
                    "memory_id": memory["id"],
                    "memory": memory,
                }

            # Same subject, refined wording — update only that memory.
            updated = self.store.update_memory(
                memory["id"],
                content=content,
                fact_type=fact_type,
                importance=importance,
                value=operation.get("value"),
                subject=subject,
                archive_history=True,
            )
            return {
                "status": "updated",
                "memory_id": memory["id"],
                "memory": updated,
                "conflict_resolved": True,
            }

        memory = self.store.add_memory(
            category=category,
            content=content,
            fact_type=fact_type,
            importance=importance,
            value=operation.get("value"),
            subject=subject,
        )
        return {"status": "added", "memory_id": memory["id"], "memory": memory}

    def _upsert_preference(
        self,
        *,
        content: str,
        importance: str,
        operation: dict,
    ) -> dict:
        new_subject = operation.get("preference_subject") or operation.get(
            "subject"
        ) or preference_subject(content)
        new_subject = normalize_preference_subject(new_subject or "")
        new_polarity = operation.get("preference_polarity") or preference_polarity(
            content
        )
        fact_type = "dislike" if new_polarity == "dislike" else "like"

        if not new_subject:
            memory = self.store.add_memory(
                category="preference",
                content=content,
                fact_type=fact_type,
                importance=importance,
                value=fact_type,
                subject=new_subject,
            )
            return {"status": "added", "memory_id": memory["id"], "memory": memory}

        for memory in self.active_memories:
            if memory.get("category") != "preference":
                continue
            old_subject = (
                memory.get("subject")
                or preference_subject(memory.get("content", ""))
            )
            if not preference_subjects_match(old_subject, new_subject):
                continue

            # Keep one canonical subject key for future comparisons.
            new_subject = normalize_preference_subject(new_subject)

            old_polarity = memory.get("fact_type")
            if old_polarity not in {"like", "dislike"}:
                old_polarity = preference_polarity(memory.get("content", ""))

            if (
                old_polarity == fact_type
                and normalized_content(memory.get("content", ""))
                == normalized_content(content)
            ):
                return {
                    "status": "unchanged",
                    "memory_id": memory["id"],
                    "memory": memory,
                }

            updated = self.store.update_memory(
                memory["id"],
                content=content,
                fact_type=fact_type,
                importance=importance,
                value=fact_type,
                subject=new_subject,
                archive_history=True,
            )
            return {
                "status": "updated",
                "memory_id": memory["id"],
                "memory": updated,
                "conflict_resolved": old_polarity != fact_type,
                "reason": (
                    "preference_reversal"
                    if old_polarity and old_polarity != fact_type
                    else "preference_refresh"
                ),
            }

        memory = self.store.add_memory(
            category="preference",
            content=content,
            fact_type=fact_type,
            importance=importance,
            value=fact_type,
            subject=new_subject,
        )
        return {"status": "added", "memory_id": memory["id"], "memory": memory}

    def _apply_goal_replacement(self, operation: dict) -> dict:
        hint = str(operation.get("old_content_hint", "") or "").strip()
        new_content = str(operation.get("content", "")).strip()
        if not new_content:
            return {"status": "skipped", "reason": "empty_content"}

        new_value = operation.get("value") or strip_goal_filler(
            re.sub(
                r"^User wants to become\s+",
                "",
                new_content,
                flags=re.IGNORECASE,
            )
        )
        fact_type = operation.get("fact_type") or "career_goal"

        # Ambiguous change (no identifiable old goal): never overwrite randomly.
        if operation.get("ambiguous") or not hint:
            return self._add_goal_safely(
                content=new_content,
                fact_type=fact_type,
                value=new_value,
                reason="ambiguous_goal_change",
            )

        target, match_reason = self._find_goal_target(hint)
        if target is None:
            return self._add_goal_safely(
                content=new_content,
                fact_type=fact_type,
                value=new_value,
                reason=f"replacement_target_{match_reason}",
            )

        updated = self.store.update_memory(
            target["id"],
            content=new_content,
            fact_type=fact_type,
            importance="high",
            value=new_value,
            subject=normalize_goal_text(new_content),
            archive_history=True,
        )
        return {
            "status": "updated",
            "memory_id": target["id"],
            "memory": updated,
            "reason": "explicit_goal_replacement",
            "conflict_resolved": True,
            "match_reason": match_reason,
        }

    def _add_goal_safely(
        self,
        *,
        content: str,
        fact_type: str,
        value: str,
        reason: str,
    ) -> dict:
        duplicate = self.find_duplicate(
            category="goal",
            fact_type=fact_type,
            content=content,
        )
        if duplicate is not None:
            if normalized_content(duplicate.get("content", "")) == normalized_content(
                content
            ) or self._semantic_same_fact(duplicate, content):
                return {
                    "status": "unchanged",
                    "memory_id": duplicate["id"],
                    "memory": duplicate,
                    "reason": reason,
                }

        memory = self.store.add_memory(
            category="goal",
            content=content,
            fact_type=fact_type,
            importance="high",
            value=value,
            subject=normalize_goal_text(content),
        )
        return {
            "status": "added",
            "memory_id": memory["id"],
            "memory": memory,
            "reason": reason,
        }

    def _apply_goal_delete(self, operation: dict) -> dict:
        hint = str(operation.get("old_content_hint", "") or "").strip()
        if not hint:
            return {"status": "skipped", "reason": "empty_goal_hint"}

        target, match_reason = self._find_goal_target(hint)
        if target is None:
            return {
                "status": "skipped",
                "reason": f"goal_not_found:{match_reason}",
            }

        memory_id = target["id"]
        self.store.delete_memory(memory_id)
        return {
            "status": "deleted",
            "memory_id": memory_id,
            "reason": "explicit_goal_delete",
            "match_reason": match_reason,
        }

    def _find_goal_target(self, hint: str) -> tuple[Optional[dict], str]:
        """
        Identify one exact goal memory for an explicit update/delete.

        Lexical match is primary. Semantic similarity may assist only when an
        explicit hint exists and exactly one active goal is a clear winner —
        never when the hint is empty/ambiguous.
        """
        hint = strip_goal_filler(hint or "")
        if not hint:
            return None, "ambiguous"

        scored: list[tuple[float, dict]] = []
        for memory in self.active_memories:
            if memory.get("category") != "goal":
                continue
            # Learning facts belong to category=learning after migration.
            if memory.get("fact_type") == "learning_goal":
                continue
            score = goal_match_score(hint, memory)
            if score >= 0.75:
                scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) == 1:
            return scored[0][1], "lexical"
        if len(scored) > 1:
            # Clear unique winner among lexical matches.
            if scored[0][0] >= scored[1][0] + 0.15:
                return scored[0][1], "lexical_best"
            return None, "ambiguous"

        # Semantic assist: explicit hint required; must be a unique clear winner
        # among active goals.  We no longer require a shared token bridge
        # between hint and memory — bare semantic similarity above the
        # threshold is sufficient when the hint is explicit.
        hint_tokens = set(normalize_goal_text(hint).split())
        if not hint_tokens:
            return None, "missing"

        probe = f"User wants to become {hint}"
        semantic = self.find_semantic_matches(
            probe,
            threshold=0.55,
            category="goal",
            fact_type="career_goal",
        )
        candidates: list[tuple[float, dict]] = []
        for score, memory in semantic:
            if str(memory.get("status", "active")).lower() != "active":
                continue
            if memory.get("fact_type") == "learning_goal":
                continue
            # Old behavior required hint_tokens & mem_tokens non-empty.
            # Removing this restriction allows semantic matches without
            # shared lexical tokens, which is exactly the relaxation requested.
            candidates.append((score, memory))

        if not candidates:
            return None, "missing"
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) == 1:
            return candidates[0][1], "semantic_assisted"
        if candidates[0][0] >= candidates[1][0] + 0.08:
            return candidates[0][1], "semantic_assisted"
        return None, "ambiguous"

    def _apply_delete_match(self, operation: dict, user_message: str) -> dict:
        if not (
            has_explicit_change_signal(user_message)
            or operation.get("explicit_change")
        ):
            return {"status": "skipped", "reason": "no_explicit_change_signal"}

        category = operation.get("category")
        fact_type = operation.get("fact_type")
        
        hints = operation.get("content_hints")
        if not hints:
            single_hint = str(operation.get("content_hint", "")).strip()
            hints = [single_hint] if single_hint else []

        if not hints:
            return {"status": "skipped", "reason": "empty_hint"}

        # Interest / learning: category-scoped target matching only.
        # Never cross into skill/goal/identity/etc.
        if category in {"interest", "learning"}:
            deleted_ids = []
            match_reasons = []
            for hint in hints:
                target, match_reason = self._find_topic_target(category, hint)
                if target is not None:
                    memory_id = target["id"]
                    if self.store.delete_memory(memory_id):
                        deleted_ids.append(memory_id)
                        match_reasons.append(match_reason)
            
            if deleted_ids:
                return {
                    "status": "deleted",
                    "memory_ids": deleted_ids,
                    "reason": f"explicit_{category}_delete",
                    "match_reason": match_reasons[0],
                }
            return {
                "status": "skipped",
                "reason": "match_not_found:missing",
            }

        for hint in hints:
            for memory in list(self.memories):
                if category and memory.get("category") != category:
                    continue
                if fact_type and memory.get("fact_type") != fact_type:
                    continue
                # Skip inactive historical snapshots.
                if str(memory.get("status", "active")).lower() != "active":
                    continue
                content = str(memory.get("content", "")).lower()
                if fact_type == "nickname" or hint.lower() == "nickname":
                    self.store.delete_memory(memory["id"])
                    return {"status": "deleted", "memory_id": memory["id"]}
                if hint and hint.lower() in content:
                    self.store.delete_memory(memory["id"])
                    return {"status": "deleted", "memory_id": memory["id"]}
                if category == "skill":
                    mem_subject = normalize_topic_subject(
                        memory.get("subject")
                        or memory.get("value")
                        or content
                    )
                    if (
                        hint
                        and mem_subject
                        and normalize_topic_subject(hint) == mem_subject
                    ):
                        self.store.delete_memory(memory["id"])
                        return {"status": "deleted", "memory_id": memory["id"]}
                if category == "preference":
                    if preference_subjects_match(
                        preference_subject(content),
                        preference_subject(f"likes {hint}"),
                    ) or preference_subjects_match(
                        memory.get("subject", ""),
                        hint,
                    ):
                        self.store.delete_memory(memory["id"])
                        return {"status": "deleted", "memory_id": memory["id"]}

        return {"status": "skipped", "reason": "match_not_found"}

    def _find_topic_target(
        self,
        category: str,
        hint: str,
    ) -> tuple[Optional[dict], str]:
        """
        Find one active memory in exactly `category` for an explicit delete.

        Lexical/spelling-normalized match is primary. Semantic similarity may
        assist only with a non-empty hint, shared tokens, and a unique winner.
        Never matches across categories (learning ≠ skill ≠ interest).
        """
        if category not in {"interest", "learning", "skill"}:
            return None, "invalid_category"
        hint_norm = normalize_topic_text(hint or "")
        if not hint_norm:
            return None, "empty_hint"

        scored: list[tuple[float, dict]] = []
        for memory in self.active_memories:
            if memory.get("category") != category:
                continue
            score = topic_match_score(hint, memory)
            if score >= 0.75:
                scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        if len(scored) == 1:
            return scored[0][1], "lexical"
        if len(scored) > 1:
            if scored[0][0] >= scored[1][0] + 0.15:
                return scored[0][1], "lexical_best"
            # If one candidate contains the hint as a full phrase, prefer it.
            for score, memory in scored:
                cand = normalize_topic_text(
                    str(memory.get("value") or memory.get("content") or "")
                )
                if hint_norm == cand:
                    return memory, "lexical_phrase"
            return None, "ambiguous"

        # Semantic assist: explicit hint + shared tokens + unique winner.
        hint_tokens = set(hint_norm.split())
        if not hint_tokens:
            return None, "missing"

        if category == "interest":
            probe = f"User is interested in {hint}"
            fact_type = "interest"
        elif category == "learning":
            probe = f"User is learning {hint}"
            fact_type = "learning"
        else:
            probe = f"User knows {hint}"
            fact_type = "skill"

        semantic = self.find_semantic_matches(
            probe,
            threshold=0.55,
            category=category,
            fact_type=fact_type,
        )
        candidates: list[tuple[float, dict]] = []
        for score, memory in semantic:
            if str(memory.get("status", "active")).lower() != "active":
                continue
            if memory.get("category") != category:
                continue
            mem_tokens = set(
                normalize_topic_text(memory.get("content", "")).split()
            )
            if not (hint_tokens & mem_tokens):
                continue
            candidates.append((score, memory))

        if not candidates:
            return None, "missing"
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) == 1:
            return candidates[0][1], "semantic_assisted"
        if candidates[0][0] >= candidates[1][0] + 0.08:
            return candidates[0][1], "semantic_assisted"
        return None, "ambiguous"

    def find_duplicate(
        self,
        *,
        category: str,
        fact_type: str,
        content: str,
    ) -> Optional[dict]:
        """Find an existing memory that represents the same logical fact."""
        new_norm = normalized_content(content)
        new_goal_norm = normalize_goal_text(content) if category == "goal" else ""

        for memory in self.active_memories:
            if memory.get("category") != category:
                continue

            # Never treat different identity fact types as duplicates.
            if category == "identity":
                if memory.get("fact_type") != fact_type:
                    continue

            # Different fact types within the same category are different facts
            # (e.g. learning_goal vs career_goal).
            existing_type = str(memory.get("fact_type", ""))
            if (
                fact_type
                and existing_type
                and fact_type != existing_type
                and category != "preference"
            ):
                continue

            old_content = str(memory.get("content", ""))

            # --- Bug 3 fix: normalized-content fallback first ---
            # Always check normalized-content match before semantic similarity.
            # This ensures duplicates are detected even when embeddings are
            # missing, None, or invalid, preventing silent "unrelated" treatment.
            if normalized_content(old_content) == new_norm:
                return memory

            if category == "goal":
                old_goal_norm = normalize_goal_text(old_content)
                if new_goal_norm and old_goal_norm == new_goal_norm:
                    return memory

            # Semantic similarity as a secondary check; safely handles
            # missing/invalid embeddings (returns False without crashing).
            if self._semantic_same_fact(memory, content):
                if category == "goal":
                    old_goal_norm = normalize_goal_text(old_content)
                    if (
                        new_goal_norm
                        and old_goal_norm
                        and new_goal_norm != old_goal_norm
                        and new_goal_norm not in old_goal_norm
                        and old_goal_norm not in new_goal_norm
                    ):
                        continue
                return memory

        return None

    def find_semantic_matches(
        self,
        content: str,
        *,
        threshold: Optional[float] = None,
        category: Optional[str] = None,
        fact_type: Optional[str] = None,
    ) -> list[tuple[float, dict]]:
        threshold = (
            threshold if threshold is not None else self.duplicate_threshold
        )
        category = category or "other"
        fact_type = fact_type or infer_fact_type(category, content)
        try:
            query_vec = self.store.embedding_store.generate_embedding(
                build_embedding_text(category, fact_type, content)
            )
        except Exception:
            return []

        scored = []
        for memory in self.memories:
            if category and memory.get("category") != category:
                continue
            emb = memory.get("embedding")
            if not isinstance(emb, list) or not emb:
                continue
            score = cosine_similarity(query_vec, emb)
            if score >= threshold:
                scored.append((score, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _semantic_same_fact(self, memory: dict, content: str) -> bool:
        emb = memory.get("embedding")
        if not isinstance(emb, list) or not emb:
            return False
        category = memory.get("category", "other")
        fact_type = memory.get("fact_type") or infer_fact_type(
            category, content
        )
        try:
            query_vec = self.store.embedding_store.generate_embedding(
                build_embedding_text(category, fact_type, content)
            )
        except Exception:
            return False
        return cosine_similarity(query_vec, emb) >= self.duplicate_threshold
