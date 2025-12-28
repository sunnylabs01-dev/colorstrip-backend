from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ASSETS_DIR = Path(__file__).parent / "assets"


def test_strips_analyze_returns_success_contract():
    image_path = ASSETS_DIR / "visiontest.jpeg"
    assert image_path.exists(), f"Missing test image: {image_path}"

    with image_path.open("rb") as f:
        res = client.post(
            "/strips/analyze",
            files={"image": ("visiontest.jpeg", f, "image/jpeg")},
        )

    assert res.status_code == 200
    data = res.json()

    # top-level contract
    assert data["ok"] is True
    assert "meta" in data
    assert "request_id" in data["meta"]
    assert isinstance(data["meta"]["request_id"], str)
    assert isinstance(data["meta"]["model_version"], str)
    assert len(data["meta"]["model_version"]) > 0

    # result contract
    result = data["result"]
    assert result["unit"] == "ppm"

    # ticks should either be both set or both None
    assert (result["lower_tick"] is None) == (result["upper_tick"] is None)

    # if ticks exist, they should be ints
    if result["lower_tick"] is not None:
        assert isinstance(result["lower_tick"], int)
        assert isinstance(result["upper_tick"], int)

    # relative_position: None or [0, 1]
    rp = result["relative_position"]
    assert (rp is None) or (0.0 <= rp <= 1.0)

    # value_ppm: None or number
    v = result["value_ppm"]
    assert (v is None) or isinstance(v, (int, float))


def test_strips_analyze_rejects_non_image_upload():
    res = client.post(
        "/strips/analyze",
        files={"image": ("not_image.txt", b"hello", "text/plain")},
    )
    body = res.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "REQ_UNSUPPORTED_MEDIA_TYPE"
    assert body["error"]["message"] == "Only image uploads are supported."
