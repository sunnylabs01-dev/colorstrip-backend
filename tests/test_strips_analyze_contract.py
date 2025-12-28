from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_analyze_contract_without_real_image_file():
    # No real image file: just bytes + image/* content-type
    res = client.post(
        "/strips/analyze",
        files={"image": ("fake.jpg", b"not-a-real-jpeg", "image/jpeg")},
    )

    assert res.status_code == 200
    data = res.json()

    # Contract: top-level
    assert set(data.keys()) >= {"ok", "meta", "result"}
    assert data["ok"] is True

    # Contract: meta
    assert set(data["meta"].keys()) >= {"request_id", "model_version"}
    assert isinstance(data["meta"]["request_id"], str)
    assert len(data["meta"]["request_id"]) > 0
    assert isinstance(data["meta"]["model_version"], str)

    # Contract: result
    result = data["result"]
    assert set(result.keys()) >= {"value_ppm", "unit", "lower_tick", "upper_tick", "relative_position"}
    assert result["unit"] == "ppm"

    # Contract: types/ranges (loose enough to survive future logic changes)
    if result["value_ppm"] is not None:
        assert isinstance(result["value_ppm"], (int, float))
    if result["lower_tick"] is not None:
        assert isinstance(result["lower_tick"], int)
    if result["upper_tick"] is not None:
        assert isinstance(result["upper_tick"], int)
    if result["relative_position"] is not None:
        assert 0.0 <= result["relative_position"] <= 1.0


def test_analyze_rejects_non_image_content_type():
    res = client.post(
        "/strips/analyze",
        files={"image": ("not_image.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "Only image uploads are supported."


def test_analyze_rejects_empty_file():
    res = client.post(
        "/strips/analyze",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "Empty file."
