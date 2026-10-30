"""Generic, extensible robot-knowledge structures.

No fixed robot-specific fields are required. New robot families can introduce
new categories and attributes without changing the retrieval provider.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KnowledgeFact:
    name: str
    value: Any
    unit: str | None = None
    description: str | None = None
    source_ids: list[str] = field(default_factory=list)
    confidence: float | None = None
    applicability: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RobotKnowledge:
    robot_id: str
    manufacturer: str | None = None
    model: str | None = None
    robot_category: str | None = None
    categories: dict[str, dict[str, Any]] = field(default_factory=dict)
    facts: list[KnowledgeFact] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "robot_category": self.robot_category,
            "categories": self.categories,
            "facts": [fact.to_dict() for fact in self.facts],
            "unknowns": self.unknowns,
            "evidence_ids": self.evidence_ids,
        }
