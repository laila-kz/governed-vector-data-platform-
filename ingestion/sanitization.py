"""Deterministic PII and secret redaction for document text."""

from __future__ import annotations

import copy
import ipaddress
import re
from dataclasses import dataclass
from typing import Any, Mapping

REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_SECRET = "[REDACTED_SECRET]"


@dataclass(frozen=True)
class Redaction:
    """One detected value and the mask used to replace it."""

    category: str
    start: int
    end: int
    mask: str


@dataclass(frozen=True)
class SanitizationResult:
    """Sanitized text and audit-friendly redaction metadata."""

    sanitized_text: str
    pii_masked_flag: bool
    redaction_counts: dict[str, int]


EMAIL_PATTERN = re.compile(
    r"(?<![\w.!#$%&'*+/=?^`{|}~-])"
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
    r"(?![\w.!#$%&'*+/=?^`{|}~-])",
    re.IGNORECASE,
)

IPV4_PATTERN = re.compile(
    r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w]|\.\d)"
)

SSN_PATTERN = re.compile(r"(?<!\d)\d{3}[- ]\d{2}[- ]\d{4}(?!\d)")
NATIONAL_ID_PATTERN = re.compile(
    r"(?i)\b(?:national[\s_-]*id|identity[\s_-]*number|tax[\s_-]*id|ssn)"
    r"\s*[:=]\s*[A-Z0-9][A-Z0-9-]{5,19}\b"
)

# These patterns intentionally require a credential marker or a known token prefix.
# That avoids treating ordinary prose and scientific numbers as secrets.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "api_key",
        re.compile(
            r"(?i)\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|secret)"
            r"\s*[:=]\s*[A-Z0-9_./+=-]{12,}"
        ),
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bBearer\s+[A-Z0-9._~+/=-]{12,}"),
    ),
    (
        "provider_token",
        re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b"),
    ),
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
)


def _is_valid_ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def _find_redactions(text: str) -> list[Redaction]:
    candidates: list[Redaction] = []
    candidates.extend(
        Redaction("email", match.start(), match.end(), REDACTED_EMAIL)
        for match in EMAIL_PATTERN.finditer(text)
    )
    candidates.extend(
        Redaction("ipv4", match.start(), match.end(), REDACTED_SECRET)
        for match in IPV4_PATTERN.finditer(text)
        if _is_valid_ipv4(match.group())
    )
    candidates.extend(
        Redaction("national_id", match.start(), match.end(), REDACTED_SECRET)
        for pattern in (SSN_PATTERN, NATIONAL_ID_PATTERN)
        for match in pattern.finditer(text)
    )
    candidates.extend(
        Redaction(category, match.start(), match.end(), REDACTED_SECRET)
        for category, pattern in SECRET_PATTERNS
        for match in pattern.finditer(text)
    )

    # Resolve overlaps deterministically: earliest match wins, then longest match.
    candidates.sort(key=lambda item: (item.start, -(item.end - item.start), item.category))
    selected: list[Redaction] = []
    for candidate in candidates:
        if selected and candidate.start < selected[-1].end:
            continue
        selected.append(candidate)
    return selected


def sanitize_text(text: str) -> SanitizationResult:
    """Replace supported PII/secrets and return an auditable result."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    redactions = _find_redactions(text)
    if not redactions:
        return SanitizationResult(text, False, {})

    pieces: list[str] = []
    counts: dict[str, int] = {}
    cursor = 0
    for redaction in redactions:
        pieces.append(text[cursor : redaction.start])
        pieces.append(redaction.mask)
        cursor = redaction.end
        counts[redaction.category] = counts.get(redaction.category, 0) + 1
    pieces.append(text[cursor:])
    return SanitizationResult("".join(pieces), True, counts)


def sanitize_document(
    document: Mapping[str, Any], text_fields: tuple[str, ...] = ("title", "text")
) -> dict[str, Any]:
    """Sanitize selected document fields and attach ``pii_masked_flag``."""
    sanitized = copy.deepcopy(dict(document))
    counts: dict[str, int] = {}
    masked = False
    for field_name in text_fields:
        value = sanitized.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"document field {field_name!r} must be a string")
        result = sanitize_text(value)
        sanitized[field_name] = result.sanitized_text
        masked = masked or result.pii_masked_flag
        for category, count in result.redaction_counts.items():
            counts[category] = counts.get(category, 0) + count

    sanitized["pii_masked_flag"] = masked
    sanitized["redaction_counts"] = counts
    return sanitized