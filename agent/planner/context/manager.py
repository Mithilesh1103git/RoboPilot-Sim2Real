"""Local-first, robot-agnostic context retrieval orchestration."""

import hashlib
import json
import logging
import os

from app.planner.context.cache import ContextCache
from app.planner.context.extractor import extract_evidence
from app.planner.context.models import PlannerContext
from app.planner.context.providers.local_profile import LocalProfileProvider
from app.planner.context.providers.tavily import TavilyProvider
from app.planner.context.validator import AUTHORITY_RULES, validate_evidence

logger = logging.getLogger(__name__)


class PlannerContextManager:
    def __init__(self, tavily_provider=None, cache=None):
        self.local_provider = LocalProfileProvider()
        self.tavily_provider = tavily_provider or TavilyProvider()
        self.cache = cache or ContextCache(
            ttl_seconds=int(os.getenv("PLANNER_CONTEXT_CACHE_TTL_SECONDS", "86400"))
        )

    @staticmethod
    def _retrieval_enabled(profile: dict | None) -> bool:
        robot_context = (profile or {}).get("robot_context") or {}
        env_enabled = os.getenv("TAVILY_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return env_enabled and bool(robot_context.get("search_enabled", False))

    @staticmethod
    def _query(profile: dict) -> str:
        cfg = profile.get("robot_context") or {}
        manufacturer = cfg.get("manufacturer") or ""
        model = (
            cfg.get("model")
            or profile.get("display_name")
            or profile.get("id")
            or "robot"
        )
        category = cfg.get("robot_category") or ""
        focus = cfg.get("search_focus") or (
            "official technical documentation, physical capabilities, kinematics, "
            "dynamics, end-effector, sensing, control interfaces and safety"
        )
        return (
            f"Find official technical documentation for {manufacturer} {model} "
            f"{category}. Research: {focus}. Prioritize manufacturer manuals and datasheets. "
            "Do not substitute specifications from other robot models."
        ).strip()

    @staticmethod
    def _cache_key(profile: dict, query: str) -> str:
        cfg = profile.get("robot_context") or {}
        material = {
            "robot_id": profile.get("id"),
            "manufacturer": cfg.get("manufacturer"),
            "model": cfg.get("model"),
            "robot_category": cfg.get("robot_category"),
            "search_focus": cfg.get("search_focus"),
            "preferred_domains": cfg.get("preferred_domains"),
            "query": query,
        }
        return hashlib.sha256(
            json.dumps(material, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def get_context(
        self, instruction: str, observation: dict | None, profile: dict | None = None
    ) -> dict:
        profile = profile or {}
        local = self.local_provider.retrieve(profile, instruction, observation)
        evidence = []
        retrieval = {"enabled": False, "status": "local_only", "provider": None}

        if self._retrieval_enabled(profile):
            retrieval["enabled"] = True
            query = self._query(profile)
            key = self._cache_key(profile, query)
            cached = self.cache.get(key)
            if cached is not None:
                evidence = cached
                retrieval.update({"status": "cache_hit", "provider": "tavily"})
            else:
                try:
                    cfg = profile.get("robot_context") or {}
                    records = self.tavily_provider.retrieve(
                        query=query,
                        preferred_domains=cfg.get("preferred_domains") or None,
                    )
                    evidence = validate_evidence(extract_evidence(records))
                    self.cache.set(key, evidence)
                    retrieval.update({"status": "retrieved", "provider": "tavily"})
                except Exception as exc:
                    logger.warning(
                        "Optional Tavily retrieval failed; using local context: %s", exc
                    )
                    retrieval.update(
                        {
                            "status": "fallback_local",
                            "provider": "tavily",
                            "error": str(exc)[:300],
                        }
                    )

        # Dynamic knowledge is intentionally evidence-first. A later structured
        # extractor can populate categories/facts without imposing Panda fields.
        robot_knowledge = {
            "robot_id": profile.get("id"),
            "manufacturer": local["robot"].get("manufacturer"),
            "model": local["robot"].get("model") or local["robot"].get("display_name"),
            "robot_category": local["robot"].get("robot_category"),
            "categories": {},
            "facts": [],
            "unknowns": [],
            "evidence_ids": [card.evidence_id for card in evidence],
        }

        context = PlannerContext(
            robot=local["robot"],
            task=local["task"],
            scene=local["scene"],
            robot_knowledge=robot_knowledge,
            evidence=evidence,
            retrieval=retrieval,
            authority_rules=list(AUTHORITY_RULES),
        )
        return context.to_dict()
