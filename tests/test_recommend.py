import pandas as pd
import pytest

from sentiment_analysis.recommend import rank_groups


def test_rank_groups_returns_sorted_scores_and_counts():
    frame = pd.DataFrame({"movie": ["A", "A", "B"]})
    predictions = [
        {"probability_positive": 0.9, "probability_negative": 0.05},
        {"probability_positive": 0.8, "probability_negative": 0.1},
        {"probability_positive": 0.2, "probability_negative": 0.7},
    ]
    result = rank_groups(frame, "movie", predictions, prior_strength=2)
    assert list(result["movie"]) == ["A", "B"]
    assert int(result.loc[result["movie"] == "A", "review_count"].iloc[0]) == 2
    assert result.loc[result["movie"] == "A", "adjusted_sentiment_score"].iloc[0] > result.loc[
        result["movie"] == "B", "adjusted_sentiment_score"
    ].iloc[0]


def test_negative_prior_strength_raises():
    with pytest.raises(ValueError, match="non-negative"):
        rank_groups(
            pd.DataFrame({"movie": ["A"]}),
            "movie",
            [{"probability_positive": 0.8, "probability_negative": 0.1}],
            prior_strength=-1,
        )
