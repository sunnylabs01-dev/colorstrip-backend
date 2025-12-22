from pydantic import BaseModel


class BaseResponse(BaseModel):
    request_id: str
    ok: bool
