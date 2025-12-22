from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ASSETS_DIR = Path(__file__).parent / "assets"


def test_strips_analyze_returns_dummy_response():
    image_path = ASSETS_DIR / "visiontest.jpeg"
    assert image_path.exists(), f"Missing test image: {image_path}"

    with image_path.open("rb") as f:
        res = client.post(
            "/v1/strips/analyze",
            files={"image": ("visiontest.jpeg", f, "image/jpeg")},
        )

    assert res.status_code == 200
    data = res.json()

    # top-level contract
    assert data["ok"] is True
    assert "meta" in data
    assert "request_id" in data["meta"]
    assert isinstance(data["meta"]["model_version"], str)
    assert len(data["meta"]["model_version"]) > 0

    # result contract
    result = data["result"]
    assert result["unit"] == "ppm"
    assert result["lower_tick"] == 40
    assert result["upper_tick"] == 50
    assert result["relative_position"] == 0.2
    assert result["value_ppm"] == 42.0


def test_strips_analyze_rejects_non_image_upload():
    res = client.post(
        "/v1/strips/analyze",
        files={"image": ("not_image.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "Only image uploads are supported."
