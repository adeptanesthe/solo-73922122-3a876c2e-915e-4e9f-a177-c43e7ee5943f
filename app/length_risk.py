"""Exact collapsed-length risk analysis over a contiguous CTC frame interval.

The beam decoder in :mod:`app.decoder` prunes to a fixed beam width, so its
candidates can never answer "what is the exact probability that this interval
collapses to *k* characters".  This module computes that distribution over
**every** CTC path without any pruning, with a forward DP whose state is the
emitted character count rather than the emitted text.

CTC length semantics, matching the decoder:

* a non-blank label emitted from a blank state (or before the first frame)
  starts a new character, i.e. increases the length by one;
* a non-blank label equal to the previous label, emitted directly without an
  intervening blank, collapses into that character and does **not** increase
  the length;
* a non-blank label different from the previous one increases the length by
  one; blank frames never change the length.

Per-character non-blank mass is kept so the "same label collapses" rule is
exact: aggregating all labels into one state would wrongly count a repeated
``a`` directly followed by another ``a`` as two characters.

States for lengths ``0 .. max_chars``:

* ``B[l]``   – log mass of paths collapsed to ``l`` characters and ending in
  blank;
* ``K[l, k]`` – log mass of paths collapsed to ``l`` characters and ending in
  non-blank label ``k``;
* ``C[l]``   – ``logsumexp`` over ``k`` of ``K[l, k]``.

Mass that has already grown beyond ``max_chars`` is folded into two aggregate
overflow states (blank / non-blank ending).  It can never return below the
limit, so character identity is no longer needed there.  The overflow mass is
exactly the over-limit probability ``P(length > max_chars)``.

A frame whose probability row is all zeros makes every continuation
impossible (every ``log p`` is ``-inf``), so no conditional distribution is
defined.  Such intervals are reported as not computable by the caller instead
of returning NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

NEG_INF = float("-inf")

MIN_INTERVAL = 1
MAX_INTERVAL = 64
MIN_LIMIT = 1
MAX_LIMIT = 16


class LengthRiskError(ValueError):
    """Invalid length-risk parameter."""


@dataclass
class LengthRiskResult:
    """Exact length distribution for one frame interval.

    ``length_logp`` / ``length_prob`` cover lengths ``0 .. max_chars``;
    ``over_limit_*`` is the aggregate mass of every longer output.
    """

    start_frame: int
    frame_count: int
    max_chars: int
    length_logp: np.ndarray
    length_prob: np.ndarray
    over_limit_logp: float
    over_limit_prob: float
    total_mass: float

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "frame_count": self.frame_count,
            "end_frame": self.start_frame + self.frame_count,
            "max_chars": self.max_chars,
            "computable": True,
            "zero_frames": [],
            "lengths": [
                {
                    "length": length,
                    "prob": float(prob),
                    "log_prob": None if logp == NEG_INF else float(logp),
                }
                for length, (prob, logp) in enumerate(
                    zip(self.length_prob, self.length_logp)
                )
            ],
            "over_limit_prob": float(self.over_limit_prob),
            "over_limit_log_prob": (
                None if self.over_limit_logp == NEG_INF else float(self.over_limit_logp)
            ),
            "total_mass": float(self.total_mass),
        }


def find_all_zero_frames(probs: np.ndarray, start: int, count: int) -> List[int]:
    """Absolute indices of all-zero probability rows inside ``[start, start+count)``.

    A row sums to zero exactly when every label (blank included) has
    probability zero, which makes the CTC conditional undefined.
    """
    block = np.asarray(probs, dtype=np.float64)[start : start + count]
    zero_mask = block.sum(axis=1) == 0.0
    return [int(start + offset) for offset in np.nonzero(zero_mask)[0]]


def length_distribution(frame_logp: np.ndarray, max_chars: int) -> LengthRiskResult:
    """Exact log/probability mass of each collapsed length, without pruning.

    ``frame_logp`` must contain natural-log row probabilities (a zero entry is
    ``-inf``) and must not include an all-zero row.  The returned probabilities
    are normalised by the total surviving path mass; when every input row is
    itself normalised (the supported case), that mass is exactly 1 and the
    values are absolute path probabilities.
    """
    frames = np.asarray(frame_logp, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[0] < 1:
        raise LengthRiskError("区间至少需要一帧")
    if np.isnan(frames).any():
        raise LengthRiskError("帧概率包含 NaN，无法计算")

    vocab_size = frames.shape[1]
    label_count = vocab_size - 1
    if label_count < 1:
        raise LengthRiskError("词表至少需要一个非 blank 字符")
    limit = int(max_chars)

    # B[l]: blank-ending mass at length l; K[l, k]: non-blank-ending mass.
    blank_state = np.full(limit + 1, NEG_INF, dtype=np.float64)
    label_state = np.full((limit + 1, label_count), NEG_INF, dtype=np.float64)
    blank_state[0] = 0.0  # empty prefix, conceptually ended in blank
    overflow_blank = NEG_INF
    overflow_label = NEG_INF

    for row in frames:
        frame_blank = float(row[0])
        frame_labels = row[1:]

        # --- blank emissions: the length never changes --------------------
        new_blank = np.logaddexp(blank_state, _label_total(label_state)) + frame_blank
        new_overflow_blank = (
            np.logaddexp(overflow_blank, overflow_label) + frame_blank
        )

        # --- non-blank emissions for lengths 1 .. limit --------------------
        new_label = np.full_like(label_state, NEG_INF)
        # Emit label k from the blank state after l-1 chars: always appends.
        from_blank = blank_state[:-1, None] + frame_labels[None, :]
        # Repeat the same label k with no blank in between: length unchanged.
        repeat = label_state[1:] + frame_labels[None, :]
        # Emit a label different from the previous one j != k: appends.
        # diff_mass[l-1, k] = log-sum-exp over j != k of K[l-1, j].
        different = _different_label_mass(label_state)[:-1] + frame_labels[None, :]
        new_label[1:] = np.logaddexp(np.logaddexp(from_blank, repeat), different)

        # --- mass crossing the limit into the overflow bucket --------------
        # From B[limit] any label appends; from K[limit, j] only a different
        # label j != k appends (repeating k stays at exactly `limit`).
        limit_diff = _different_label_mass(label_state)[limit]
        cross_per_label = (
            np.logaddexp(blank_state[limit], limit_diff) + frame_labels
        )
        limit_cross = np.logaddexp.reduce(cross_per_label)
        # Once over the limit, blank or any label keeps the path over it.
        overflow_any_label = (
            np.logaddexp(overflow_blank, overflow_label)
            + np.logaddexp.reduce(frame_labels)
        )
        new_overflow_label = np.logaddexp(overflow_any_label, limit_cross)

        blank_state = new_blank
        label_state = new_label
        overflow_blank = new_overflow_blank
        overflow_label = new_overflow_label

    length_logp = np.logaddexp(blank_state, _label_total(label_state))
    overflow_logp = np.logaddexp(overflow_blank, overflow_label)

    all_logp = np.concatenate([length_logp, np.array([overflow_logp])])
    finite = all_logp[np.isfinite(all_logp)]
    if finite.size == 0:
        # Every path died (e.g. each frame forces an impossible transition).
        # Treat this like the all-zero case: no conditional distribution.
        raise LengthRiskError("区间内不存在概率非零的路径，无法计算")
    total_logp = np.logaddexp.reduce(finite)
    total_mass = float(np.exp(total_logp))

    length_prob = np.exp(length_logp - total_logp)
    over_limit_prob = (
        0.0 if overflow_logp == NEG_INF else float(np.exp(overflow_logp - total_logp))
    )
    return LengthRiskResult(
        start_frame=0,
        frame_count=int(frames.shape[0]),
        max_chars=limit,
        length_logp=length_logp,
        length_prob=length_prob,
        over_limit_logp=overflow_logp,
        over_limit_prob=over_limit_prob,
        total_mass=total_mass,
    )


def _label_total(label_state: np.ndarray) -> np.ndarray:
    """Per-length log-sum-exp over the non-blank label dimension."""
    return np.logaddexp.reduce(label_state, axis=1)


def _different_label_mass(label_state: np.ndarray) -> np.ndarray:
    """For each length ``l`` and label ``k``: log-sum-exp over ``j != k``.

    This is the incoming non-blank mass that appends a new character, because
    a label directly following the *same* label without an intervening blank
    collapses instead.  With a single non-blank label the result is always
    ``-inf``: nothing is different from it.
    """
    label_count = label_state.shape[1]
    if label_count <= 1:
        return np.full_like(label_state, NEG_INF)
    mass = label_state[:, :, None]  # (lengths, previous-label j, 1)
    not_same = ~np.eye(label_count, dtype=bool)[None, :, :]  # j != k mask
    masked = np.where(not_same, mass, NEG_INF)
    return np.logaddexp.reduce(masked, axis=1)


def validate_interval(start_frame: int, frame_count: int, max_chars: int) -> tuple[int, int, int]:
    """Normalise and bounds-check the risk-analysis parameters."""
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise LengthRiskError("起始帧必须是整数")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise LengthRiskError("区间帧数必须是整数")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise LengthRiskError("字数上限必须是整数")
    if start_frame < 0:
        raise LengthRiskError("起始帧不能为负")
    if not MIN_INTERVAL <= frame_count <= MAX_INTERVAL:
        raise LengthRiskError(f"区间帧数必须在 {MIN_INTERVAL}～{MAX_INTERVAL} 之间")
    if not MIN_LIMIT <= max_chars <= MAX_LIMIT:
        raise LengthRiskError(f"字数上限必须在 {MIN_LIMIT}～{MAX_LIMIT} 之间")
    return start_frame, frame_count, max_chars


def compute_length_risk(
    probs: np.ndarray,
    frame_logp: np.ndarray,
    start_frame: int,
    frame_count: int,
    max_chars: int,
    advanced_frames: int,
) -> Optional[dict]:
    """Compute the JSON payload for an interval of already-advanced frames.

    When the interval contains an all-zero frame the payload is a normal
    ``computable: false`` response (with the offending frame indices) instead
    of an exception, so the endpoint can return 200 and the client keeps its
    previously displayed result.
    """
    start_frame, frame_count, max_chars = validate_interval(
        start_frame, frame_count, max_chars
    )
    total_frames = int(np.asarray(probs).shape[0])
    if start_frame + frame_count > total_frames:
        raise LengthRiskError("区间超出音频总帧数")
    if start_frame + frame_count > advanced_frames:
        raise LengthRiskError("只能分析已经推进过的连续帧")

    zero_frames = find_all_zero_frames(probs, start_frame, frame_count)
    if zero_frames:
        return {
            "start_frame": start_frame,
            "frame_count": frame_count,
            "end_frame": start_frame + frame_count,
            "max_chars": max_chars,
            "computable": False,
            "zero_frames": zero_frames,
            "message": (
                "区间内第 "
                + "、".join(str(index) for index in zero_frames)
                + " 帧概率全为 0，条件概率无定义，无法计算；已保留上次分析结果。"
            ),
        }

    block = np.asarray(frame_logp, dtype=np.float64)[start_frame : start_frame + frame_count]
    result = length_distribution(block, max_chars)
    result.start_frame = start_frame
    result.frame_count = frame_count
    return result.to_dict()
