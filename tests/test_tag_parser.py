"""Tests for the client-side tag parser."""

from ataf.client.policy import (
    DECLINE_NO_AUTHORIZED_TOOL,
    DECLINE_NOT_CAPABLE,
    DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED,
)
from ataf.client.tag_parser import (
    classify_text_response,
    extract_ataf_code,
    find_decline_code,
)


PROPOSAL = '''Sure, here is a tool:

<ATAF param="newtool">
def circle_area(radius: float) -> float:
    """Compute the area of a circle.

    Args:
        radius: The radius.
    """
    import math
    return math.pi * radius ** 2
</ATAF>

Please deploy it.
'''


def test_extract_code_returns_function_body() -> None:
    """The code inside the ATAF tag is extracted, trimmed of blank lines."""

    code = extract_ataf_code(PROPOSAL)
    assert code is not None
    assert code.startswith("def circle_area(radius: float) -> float:")
    assert code.rstrip().endswith("return math.pi * radius ** 2")


def test_extract_code_none_when_absent() -> None:
    """No tag -> None."""

    assert extract_ataf_code("just some prose") is None


def test_classify_proposal() -> None:
    """A response with an ATAF tag classifies as propose."""

    parsed = classify_text_response(PROPOSAL)
    assert parsed.kind == "propose"
    assert parsed.code is not None


def test_classify_decline() -> None:
    """A response with a decline code classifies as decline."""

    parsed = classify_text_response(
        f"{DECLINE_NOT_CAPABLE}: I cannot write that tool."
    )
    assert parsed.kind == "decline"
    assert parsed.decline_code == DECLINE_NOT_CAPABLE


def test_classify_prose() -> None:
    """Plain text with no tag or code classifies as prose."""

    parsed = classify_text_response("The area is about 380.13.")
    assert parsed.kind == "prose"
    assert parsed.code is None
    assert parsed.decline_code is None


def test_proposal_beats_decline_text() -> None:
    """If both a tag and a decline-looking word appear, propose wins."""

    text = PROPOSAL + f"\n(otherwise {DECLINE_NOT_CAPABLE})"
    assert classify_text_response(text).kind == "propose"


def test_find_decline_code_each_variant() -> None:
    """Each of the three decline codes is recognized."""

    for code in (
        DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED,
        DECLINE_NOT_CAPABLE,
        DECLINE_NO_AUTHORIZED_TOOL,
    ):
        assert find_decline_code(f"Response: {code}") == code


def test_find_decline_code_none_for_prose() -> None:
    """Prose without a code yields None."""

    assert find_decline_code("I think the answer is 42.") is None
