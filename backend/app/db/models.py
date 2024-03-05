from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
)
from sqlalchemy.sql import func
from app.db.database import Base
from sqlalchemy.orm import relationship


class Member(Base):
    __tablename__ = "member"
    member_id = Column(
        Integer, primary_key=True, autoincrement=True, index=True
    )
    email = Column(String, unique=True, index=True)
    password = Column(String)
    member_name = Column(String)
    store_name = Column(String)
    create_time = Column(DateTime(timezone=True), server_default=func.now())
    update_time = Column(DateTime(timezone=True), onupdate=func.now())
    threshold = Column(Float, default=0.8)
    save_time_length = Column(Integer, default=60)
    cctv_info_yn = Column(Boolean, default=False)

    cctv = relationship("CCTV", back_populates="member")


class CCTV(Base):
    __tablename__ = "cctv"
    cctv_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    member_id = Column(Integer, ForeignKey(Member.member_id))
    cctv_name = Column(String)
    cctv_url = Column(String)

    member = relationship("Member", back_populates="cctv")


class Log(Base):
    __tablename__ = "log"
    log_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    cctv_id = Column(Integer, ForeignKey(CCTV.cctv_id))
    anomaly_create_time = Column(
        DateTime(timezone=True), server_default=func.now()
    )
    anomaly_score = Column(Float)
    anomaly_save_path = Column(String)
    anomaly_feedback = Column(Integer, default=-1)
    anomaly_delete_yn = Column(Boolean, default=False)
