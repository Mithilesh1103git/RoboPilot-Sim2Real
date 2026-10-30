"""Normalized planner-context and evidence models."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceCard:
    evidence_id: str
    title: str
    url: str
    snippet: str
    source: str = "tavily"
    score: float | None = None
    retrieved_at: str | None = None
    published_date: str | None = None
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerContext:
    robot: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    scene: dict[str, Any] = field(default_factory=dict)
    robot_knowledge: dict[str, Any] = field(default_factory=dict)
    evidence: list[EvidenceCard] = field(default_factory=list)
    retrieval: dict[str, Any] = field(default_factory=dict)
    authority_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "task": self.task,
            "scene": self.scene,
            "robot_knowledge": self.robot_knowledge,
            "evidence": [item.to_dict() for item in self.evidence],
            "retrieval": self.retrieval,
            "authority_rules": self.authority_rules,
        }
