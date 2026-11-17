"""HTTP tests for the length-risk endpoint.

The key guarantees checked here:

* the endpoint is read-only: beams, prefix, progress and revision are
  identical before and after an analysis, including alongside confirm/undo;
* all-zero frames return ``computable: false`` instead of an error, so the
  client can retain its previous result;
* parameter bounds and the "advanced frames only" restriction are enforced;
* the response revision lets the client tell stale answers from current state.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

VOCAB = ["<blank>", "a", "b"]
FRAMES = [
    [0.6, 0.4, 0.0],
    [0.1, 0.8, 0.1],
    [0.5, 0.0, 0.5],
    [0.2, 0.3, 0.5],
    [0.7, 0.2, 0.1],
]
# One fully zero row: every label has probability zero.
ZERO_FRAMES = [row[:] for row in FRAMES] + [[0.0, 0.0, 0.0], [0.5, 0.3, 0.2]]


@pytest.fixture()
def client():
    return TestClient(app)


def create(client, frames=FRAMES):
    response = client.post(
        "/api/sessions",
        json={"vocab": VOCAB, "beam_width": 4, "frames": frames},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_length_risk_basic_shape_and_probabilities(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})

    response = client.post(
        "/api/length-risk",
        json={"session_id": sid, "start_frame": 0, "frame_count": 5, "max_chars": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["computable"] is True
    assert payload["start_frame"] == 0 and payload["frame_count"] == 5 and payload["end_frame"] == 5
    assert payload["max_chars"] == 2
    assert [row["length"] for row in payload["lengths"]] == [0, 1, 2]
    total = sum(row["prob"] for row in payload["lengths"]) + payload["over_limit_prob"]
    assert total == pytest.approx(1.0, abs=1e-12)
    assert payload["total_mass"] == pytest.approx(1.0, abs=1e-12)
    # the response carries the current revision for staleness checks
    assert payload["revision"] == client.get(f"/api/sessions/{sid}").json()["revision"]


def test_length_risk_accepts_arbitrary_advanced_subinterval(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    response = client.post(
        "/api/length-risk",
        json={"session_id": sid, "start_frame": 2, "frame_count": 3, "max_chars": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["start_frame"] == 2 and payload["end_frame"] == 5
    assert [row["length"] for row in payload["lengths"]] == [0, 1]


def test_length_risk_is_read_only_alongside_confirm_and_undo(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    before = client.get(f"/api/sessions/{sid}").json()

    response = client.post(
        "/api/length-risk",
        json={"session_id": sid, "start_frame": 0, "frame_count": 3, "max_chars": 2},
    )
    assert response.status_code == 200
    after = client.get(f"/api/sessions/{sid}").json()
    # Nothing in the decoding session changed: revision, beams, progress, prefix.
    assert after["revision"] == before["revision"]
    assert after["beams"] == before["beams"]
    assert after["next_frame"] == before["next_frame"] == 3
    assert after["prefix"] == before["prefix"] == ""

    target = before["beams"][0]["text"]
    client.post("/api/confirm", json={"session_id": sid, "prefix": target})
    confirmed = client.get(f"/api/sessions/{sid}").json()
    client.post(
        "/api/length-risk",
        json={"session_id": sid, "start_frame": 0, "frame_count": 3, "max_chars": 2},
    )
    after_risk = client.get(f"/api/sessions/{sid}").json()
    assert after_risk["revision"] == confirmed["revision"]
    assert after_risk["prefix"] == target
    assert after_risk["beams"] == confirmed["beams"]

    client.post("/api/undo", json={"session_id": sid})
    undone = client.get(f"/api/sessions/{sid}").json()
    assert undone["prefix"] == ""
    assert undone["beams"] == before["beams"]


def test_length_risk_rejects_unadvanced_interval(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    response = client.post(
        "/api/length-risk",
        json={"session_id": sid, "start_frame": 0, "frame_count": 3, "max_chars": 2},
    )
    assert response.status_code == 400
    assert "推进" in response.json()["detail"]
    # failed analysis leaves the session untouched
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["next_frame"] == 2


def test_length_risk_parameter_validation(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    for body in (
        {"start_frame": 0, "frame_count": 0, "max_chars": 2},
        {"start_frame": 0, "frame_count": 65, "max_chars": 2},
        {"start_frame": 0, "frame_count": 2, "max_chars": 0},
        {"start_frame": 0, "frame_count": 2, "max_chars": 17},
        {"start_frame": -1, "frame_count": 2, "max_chars": 2},
        {"start_frame": 4, "frame_count": 2, "max_chars": 2},
    ):
        response = client.post(
            "/api/length-risk", json={"session_id": sid, **body}
        )
        assert response.status_code == 400, body


def test_length_risk_unknown_session_is_404(client):
    response = client.post(
        "/api/length-risk",
        json={"session_id": "missing", "start_frame": 0, "frame_count": 1, "max_chars": 1},
    )
    assert response.status_code == 404


def test_all_zero_frame_returns_computable_false(client):
    session = create(client, frames=ZERO_FRAMES)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 7})

    # interval covering the all-zero frame (index 5): not computable
    response = client.post(
        "/api/length-risk",
        json={"session_id": sid, "start_frame": 3, "frame_count": 4, "max_chars": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["computable"] is False
    assert payload["zero_frames"] == [5]
    assert "无法计算" in payload["message"]

    # the same interval ending just before the zero frame is computable
    response = client.post(
        "/api/length-risk",
        json={"session_id": sid, "start_frame": 3, "frame_count": 2, "max_chars": 2},
    )
    assert response.json()["computable"] is True

    # the session is untouched by the uncomputable response
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["next_frame"] == 7


def test_revision_changes_only_with_mutating_ops(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    revision = client.get(f"/api/sessions/{sid}").json()["revision"]

    payload = client.post(
        "/api/length-risk",
        json={"session_id": sid, "start_frame": 0, "frame_count": 3, "max_chars": 2},
    ).json()
    assert payload["revision"] == revision

    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    newer = client.get(f"/api/sessions/{sid}").json()["revision"]
    assert newer > revision
    # an in-flight-style late answer would carry the old revision; the client
    # uses this field to refuse applying it over the newer state
    assert revision < newer
