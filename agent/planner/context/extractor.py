"""Robot-agnostic evidence normalization.

This module deliberately does not assume every robot has payload, reach,
joint limits, a gripper, or locomotion. It preserves source evidence so a
later structured extraction step can identify only supported attributes.
"""

import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse

from app.planner.context.models import EvidenceCard


def extract_evidence(records: list[dict]) -> list[EvidenceCard]:
    cards = []
    for record in records:
        url = str(record.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue

        snippet = " ".join(
            str(record.get("content") or record.get("snippet") or "").split()
        )
        if not snippet:
            continue

        title = str(record.get("title") or parsed.netloc)[:300]
        evidence_id = hashlib.sha256(f"{url}\n{title}".encode("utf-8")).hexdigest()[:16]

        try:
            score = float(record["score"]) if record.get("score") is not None else None
        except (TypeError, ValueError):
            score = None

        cards.append(
            EvidenceCard(
                evidence_id=evidence_id,
                title=title,
                url=url,
                snippet=snippet[:2500],
                source=str(record.get("source") or "unknown"),
                score=score,
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                published_date=record.get("published_date"),
                domain=parsed.netloc.lower(),
            )
        )
    return cards
