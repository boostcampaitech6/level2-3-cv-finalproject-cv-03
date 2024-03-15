from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class CustomBaseModel(BaseModel):
    class Config:
        orm_mode = True


class Login(BaseModel):
    email: str
    password: str


class MemberBase(CustomBaseModel):
    email: str
    password: str
    member_name: Optional[str]
    store_name: str


class MemberDetail(MemberBase):
    member_id: int
    threshold: float
    save_time_length: int
    cctv_info_yn: bool
    create_time: datetime
    update_time: Optional[datetime]


class CCTVBase(CustomBaseModel):
    cctv_id: int
    member_id: int
    cctv_name: str
    cctv_url: str
    cctv_delete_yn: bool


class LogBase(CustomBaseModel):
    log_id: int
    cctv_id: int
    anomaly_score: float
    anomaly_save_path: str
    anomaly_feedback: int
    anomaly_delete_yn: bool
