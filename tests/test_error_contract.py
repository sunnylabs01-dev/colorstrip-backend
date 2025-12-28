from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_request_id_header_present_on_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert "x-request-id" in res.headers
    assert res.headers["x-request-id"]


def test_validation_error_returns_error_response_schema():
    # 파일 없이 호출 -> 422
    res = client.post("/strips/analyze")
    assert res.status_code == 422

    body = res.json()
    assert body["ok"] is False
    assert "request_id" in body
    assert "error" in body
    assert body["error"]["code"] == "REQ_VALIDATION_FAILED"
    assert isinstance(body["error"].get("details", {}).get("field_errors", []), list)

    # header/body request_id 일치
    assert res.headers.get("x-request-id") == body["request_id"]


def test_app_error_returns_error_response_schema():
    # debug route는 400을 발생시키도록 만들어둔 상태
    res = client.get("/debug/error")
    assert res.status_code == 400

    body = res.json()
    assert body["ok"] is False
    assert "request_id" in body
    assert body["error"]["code"]  # REQ_DEBUG_ERROR
    assert res.headers.get("x-request-id") == body["request_id"]
