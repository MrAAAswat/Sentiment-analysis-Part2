import pytest

from sentiment_analysis.data import normalize_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("negative", "negative"),
        (" NEG ", "negative"),
        ("neutral", "neutral"),
        ("positive", "positive"),
        ("pos", "positive"),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="Unsupported sentiment label"):
        normalize_label("mixed")
