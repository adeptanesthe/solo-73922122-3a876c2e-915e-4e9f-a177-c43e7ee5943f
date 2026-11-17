"""Tests for the exact collapsed-length risk analysis.

The distribution is cross-checked against a full enumeration of every CTC
label path for tiny frames, mirroring the brute-force approach used for the
beam decoder.  No pruning may be involved: these tests would fail if the
analysis reused the decoder's beam candidates.
"""

import itertools
import math

import numpy as np
import pytest

from app.length_risk import (
    MAX_INTERVAL,
    MAX_LIMIT,
    LengthRiskError,
    compute_length_risk,
    find_all_zero_frames,
    length_distribution,
)


def log_probs(frames):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(frames, dtype=np.float64))


def brute_lengths(probs, max_chars):
    """Enumerate all paths; return ({length: prob}, over_limit_prob, total)."""
    v_size = len(probs[0])
    buckets = {}
    total = 0.0
    for path in itertools.product(range(v_size), repeat=len(probs)):
        logp = 0.0
        alive = True
        for t, k in enumerate(path):
            p = probs[t][k]
            if p <= 0.0:
                alive = False
                break
            logp += math.log(p)
        if not alive:
            continue
        length = 0
        previous = 0
        for k in path:
            if k != 0 and k != previous:
                length += 1
            previous = k
        mass = math.exp(logp)
        buckets[length] = buckets.get(length, 0.0) + mass
        total += mass
    dist = {length: mass / total for length, mass in buckets.items() if length <= max_chars}
    over = sum(mass for length, mass in buckets.items() if length > max_chars) / total
    return dist, over, total


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("limit", [1, 2, 3, 6])
def test_distribution_matches_brute_force(seed, limit):
    rng = np.random.default_rng(seed)
    v_size = int(rng.integers(2, 5))
    t_size = int(rng.integers(1, 8))
    probs = rng.random((t_size, v_size))
    if seed % 4 == 0:
        # a hard zero somewhere: some paths become impossible
        probs[int(rng.integers(0, t_size)), int(rng.integers(0, v_size))] = 0.0
    probs /= probs.sum(axis=1, keepdims=True)

    result = length_distribution(log_probs(probs), limit)
    expected, expected_over, total = brute_lengths(probs.tolist(), limit)

    assert result.length_prob.shape == (limit + 1,)
    for length in range(limit + 1):
        assert result.length_prob[length] == pytest.approx(expected.get(length, 0.0), abs=1e-12)
    assert result.over_limit_prob == pytest.approx(expected_over, abs=1e-12)
    assert result.length_prob.sum() + result.over_limit_prob == pytest.approx(1.0, abs=1e-12)
    assert result.total_mass == pytest.approx(total, abs=1e-12)
    # log / probability consistency against the absolute path mass
    for length in range(limit + 1):
        if result.length_prob[length] > 0:
            assert math.exp(result.length_logp[length]) == pytest.approx(
                result.length_prob[length] * total, abs=1e-12
            )


def test_repeated_label_without_blank_counts_once():
    # All frames force "a": any number of frames collapses to exactly one char.
    probs = [[0.0, 1.0, 0.0]] * 5
    result = length_distribution(log_probs(probs), 4)
    assert result.length_prob[1] == pytest.approx(1.0)
    assert result.length_prob[0] == 0.0
    assert result.over_limit_prob == 0.0
    # and the same holds at the smallest possible limit
    result = length_distribution(log_probs(probs), 1)
    assert result.length_prob[1] == pytest.approx(1.0)


def test_repeat_needs_blank_to_grow():
    # a, blank, a, blank, a -> exactly "aaa"; without the blanks it would be "a".
    probs = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    result = length_distribution(log_probs(probs), 4)
    assert result.length_prob[3] == pytest.approx(1.0)

    result = length_distribution(log_probs(probs), 2)
    assert result.over_limit_prob == pytest.approx(1.0)
    assert result.length_prob.sum() == 0.0
    result = length_distribution(log_probs(probs), 1)
    assert result.over_limit_prob == pytest.approx(1.0)


def test_overflow_bucket_keeps_exact_mass():
    # Three non-blank frames with blanks between: limit 1 must collect every
    # path whose collapsed output is longer than 1, without dropping any.
    probs = [
        [0.2, 0.7, 0.1],
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
    ]
    for limit in (1, 2, 3):
        result = length_distribution(log_probs(probs), limit)
        expected, expected_over, _ = brute_lengths(probs, limit)
        for length in range(limit + 1):
            assert result.length_prob[length] == pytest.approx(expected.get(length, 0.0), abs=1e-12)
        assert result.over_limit_prob == pytest.approx(expected_over, abs=1e-12)


def test_non_normalised_frames_normalise_by_total_mass():
    # Rows summing below 1 shrink the surviving mass; the distribution must
    # still be a proper conditional over surviving paths.
    rng = np.random.default_rng(3)
    probs = rng.random((5, 3)) * 0.4
    result = length_distribution(log_probs(probs), 4)
    assert result.total_mass == pytest.approx(float(np.prod(probs.sum(axis=1))), rel=1e-12)
    assert result.length_prob.sum() + result.over_limit_prob == pytest.approx(1.0, abs=1e-12)


def test_all_zero_frame_is_flagged_and_kept_separate():
    probs = np.array([[0.5, 0.5], [0.0, 0.0], [0.5, 0.5]])
    assert find_all_zero_frames(probs, 0, 3) == [1]
    assert find_all_zero_frames(probs, 0, 1) == []
    assert find_all_zero_frames(probs, 2, 1) == []
    with np.errstate(divide="ignore"):
        frames = np.log(probs)
    payload = compute_length_risk(probs, frames, 0, 3, 1, advanced_frames=3)
    assert payload["computable"] is False
    assert payload["zero_frames"] == [1]
    assert "无法计算" in payload["message"] and "保留" in payload["message"]
    # sub-intervals that avoid the zero frame remain computable
    assert compute_length_risk(probs, frames, 0, 1, 1, advanced_frames=3)["computable"]
    assert compute_length_risk(probs, frames, 2, 1, 1, advanced_frames=3)["computable"]


def test_wrapper_rejects_unadvanced_and_out_of_range_intervals():
    probs = np.full((10, 3), 1.0 / 3)
    with np.errstate(divide="ignore"):
        frames = np.log(probs)
    with pytest.raises(LengthRiskError):
        compute_length_risk(probs, frames, 0, 4, 1, advanced_frames=3)
    with pytest.raises(LengthRiskError):
        compute_length_risk(probs, frames, 8, 3, 1, advanced_frames=10)
    with pytest.raises(LengthRiskError):
        compute_length_risk(probs, frames, -1, 1, 1, advanced_frames=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(start_frame=0, frame_count=0, max_chars=1),
        dict(start_frame=0, frame_count=MAX_INTERVAL + 1, max_chars=1),
        dict(start_frame=0, frame_count=1, max_chars=0),
        dict(start_frame=0, frame_count=1, max_chars=MAX_LIMIT + 1),
        dict(start_frame=True, frame_count=1, max_chars=1),
        dict(start_frame=0, frame_count=1.5, max_chars=1),
    ],
)
def test_parameter_bounds(kwargs):
    probs = np.full((80, 2), 0.5)
    with np.errstate(divide="ignore"):
        frames = np.log(probs)
    with pytest.raises(LengthRiskError):
        compute_length_risk(probs, frames, advanced_frames=80, **kwargs)


def test_payload_shape_and_bounds():
    probs = np.full((6, 3), 1.0 / 3)
    with np.errstate(divide="ignore"):
        frames = np.log(probs)
    payload = compute_length_risk(probs, frames, 1, 4, 2, advanced_frames=6)
    assert payload["computable"] is True
    assert payload["start_frame"] == 1
    assert payload["frame_count"] == 4
    assert payload["end_frame"] == 5
    assert payload["max_chars"] == 2
    assert [row["length"] for row in payload["lengths"]] == [0, 1, 2]
    assert all(0.0 <= row["prob"] <= 1.0 for row in payload["lengths"])
    assert 0.0 <= payload["over_limit_prob"] <= 1.0
    assert payload["total_mass"] == pytest.approx(1.0, abs=1e-12)
    total = sum(row["prob"] for row in payload["lengths"]) + payload["over_limit_prob"]
    assert total == pytest.approx(1.0, abs=1e-12)


def test_analysis_does_not_depend_on_beam_width():
    # The exact distribution must be identical regardless of how narrow the
    # decoding beam is: it never reads beam candidates.
    from app.state import SessionManager

    frames = [
        [0.15, 0.7, 0.15],
        [0.7, 0.2, 0.1],
        [0.15, 0.15, 0.7],
        [0.7, 0.2, 0.1],
        [0.15, 0.7, 0.15],
    ]
    manager = SessionManager()
    narrow = manager.create(["<blank>", "a", "b"], 1, frames)
    wide = manager.create(["<blank>", "a", "b"], 16, frames)
    payloads = []
    for session in (narrow, wide):
        payloads.append(
            compute_length_risk(
                session.probs, session.frames, 0, 5, 3, advanced_frames=5
            )
        )
    assert payloads[0]["lengths"] == payloads[1]["lengths"]
    assert payloads[0]["over_limit_prob"] == payloads[1]["over_limit_prob"]
