"""Tag parser — classify the LLM's *text* output into an ATAF response.

The LLM has up to four response shapes (DESIGN.md §3). Two of them are
expressed in the model's native tool-use channel and are handled by the
LLM adapter (a ``tool_use`` block for invocation). The other two —
**proposing a new tool** (``<ATAF param="newtool">…</ATAF>``) and
**declining** (a ``CANNOT_IMPLEMENT_*`` reason code) — come through as
plain text. This module recognizes those, and treats anything else as a
**prose** answer.

It is deliberately small and regex-based: the proposal tag and the decline
codes are distinctive enough that a full parser would be overkill.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .policy import DECLINE_CODES


# Matches a <ATAF ...>...</ATAF> block. We allow any attributes on the open
# tag (the canonical form is param="newtool") and capture the inner body
# non-greedily so multiple tags don't merge. DOTALL so the code can span
# newlines; IGNORECASE so <ataf>/<ATAF> both work.
_ATAF_BLOCK = re.compile(
    r"<ATAF\b[^>]*>(.*?)</ATAF\s*>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedResponse:
    """The classified result of one LLM text response.

    Exactly one of the three kinds applies, indicated by ``kind``:

    Attributes:
        kind: ``"propose"``, ``"decline"``, or ``"prose"``.
        code: The proposed Python source (only when kind is ``"propose"``).
        decline_code: The ``CANNOT_IMPLEMENT_*`` reason (only when kind is
            ``"decline"``).
        text: The model's full text, preserved for logging and for use as
            the final answer when kind is ``"prose"``.
    """

    kind: str
    code: str | None
    decline_code: str | None
    text: str


def extract_ataf_code(text: str) -> str | None:
    """Return the Python source inside the first ``<ATAF>`` block, if any.

    Args:
        text: The LLM's raw text output.

    Returns:
        The de-indented tool source with surrounding blank lines trimmed,
        or None if there is no ``<ATAF>`` block.
    """

    match = _ATAF_BLOCK.search(text)
    if match is None:
        return None

    # Strip only leading/trailing blank lines; preserve the code's own
    # indentation so the builder sees exactly what the LLM wrote.
    body = match.group(1)
    return body.strip("\n").rstrip() + "\n"


def find_decline_code(text: str) -> str | None:
    """Return the decline reason code present in the text, if any.

    Args:
        text: The LLM's raw text output.

    Returns:
        The matched ``CANNOT_IMPLEMENT_*`` code, or None. If more than one
        appears (the model shouldn't do that), the first by source order
        wins.
    """

    # The codes are distinctive uppercase tokens; a bounded search avoids
    # matching a code that appears as a substring of a longer identifier.
    best_index = None
    best_code = None
    for code in DECLINE_CODES:
        match = re.search(rf"\b{re.escape(code)}\b", text)
        if match is not None and (best_index is None or match.start() < best_index):
            best_index = match.start()
            best_code = code
    return best_code


def classify_text_response(text: str) -> ParsedResponse:
    """Classify an LLM text response into propose / decline / prose.

    Priority order:
      1. **propose** — an ``<ATAF>`` block is present (it carries
         actionable code, so it wins even if other text surrounds it).
      2. **decline** — a ``CANNOT_IMPLEMENT_*`` code is present.
      3. **prose** — anything else (a direct natural-language answer).

    Args:
        text: The LLM's raw text output.

    Returns:
        A ``ParsedResponse`` describing the recognized shape.
    """

    # 1. A proposal beats everything else — extract and return the code.
    code = extract_ataf_code(text)
    if code is not None:
        return ParsedResponse(kind="propose", code=code, decline_code=None, text=text)

    # 2. A structured decline code.
    decline_code = find_decline_code(text)
    if decline_code is not None:
        return ParsedResponse(
            kind="decline", code=None, decline_code=decline_code, text=text
        )

    # 3. Otherwise it's a plain prose answer.
    return ParsedResponse(kind="prose", code=None, decline_code=None, text=text)
