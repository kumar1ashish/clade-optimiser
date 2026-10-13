"""The support-desk test split agrees with its written policy, and its hidden tests tell right from wrong."""

from bench.benchmarks.support_desk.validate import check


def test_support_desk_test_split_is_valid():
    rows, problems = check()
    assert len(rows) == 20 and all(r["test"] for r in rows)
    assert problems == []
    assert sum(r["checks"]["fails the careless agent"] is True for r in rows) == 15  # most tasks need the policy
