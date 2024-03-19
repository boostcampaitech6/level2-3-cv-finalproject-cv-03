import os
import ssl
import json
import redis
import random
import shutil
import string
import smtplib

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

from app.db.database import get_db
from app.schemas import Login
from app.db import models

from passlib.context import CryptContext
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

DEFAULT_RETURN_DICT = {"isSuccess": True, "result": None}

SMTP_SSL_PORT = 465  # SSL connection
SMTP_SERVER = "smtp.gmail.com"

SENDER_EMAIL = "gusdn00751@gmail.com"
SENDER_PASSWORD = "wcrcavozgdpcehwd"


router = APIRouter()
memberRouter = APIRouter(prefix="/api/v0/members")
cctvRouter = APIRouter(prefix="/api/v0/cctv")
streamingRouter = APIRouter(prefix="/api/v0/streaming")
settingRouter = APIRouter(prefix="/api/v0/settings")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

redis_server = redis.Redis(host="10.28.224.201", port=30575, db=0)


def hash_password(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def generate_verification_code():
    characters = string.ascii_letters + string.digits
    code = "".join(random.choice(characters) for i in range(8))
    return code


@memberRouter.get("/send_auth")
def send_auth(email: str, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    code = generate_verification_code()
    auth = models.EmailAuth(
        email=email,
        code=code,
    )

    context = ssl.create_default_context()

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = email
    msg["Subject"] = "와치덕 인증번호입니다."

    email_content = """
    <html>
        <body>
            <p>안녕하세요.<br>
            인증코드는 {code}입니다.<br>
            감사합니다.
            </p>
        </body>
    </html>
    """.format(
        code=code
    )

    msg.attach(MIMEText(email_content, "html"))

    try:
        with smtplib.SMTP_SSL(
            SMTP_SERVER, SMTP_SSL_PORT, context=context
        ) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, msg.as_string())
    except Exception:
        def_return_dict["isSuccess"] = False
        return def_return_dict
    temp = (
        session.query(models.EmailAuth)
        .filter(models.EmailAuth.email == email)
        .first()
    )

    if temp:
        temp.code = code

    else:
        session.add(auth)

    session.commit()

    return def_return_dict


@memberRouter.get("/confirm_auth")
def confirm_auth(email: str, code: str, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    temp = (
        session.query(models.EmailAuth)
        .filter(models.EmailAuth.email == email)
        .first()
    )

    if temp:
        if temp.code == code:
            session.delete(temp)
            session.commit()
            return def_return_dict
        else:
            def_return_dict["isSuccess"] = False
            return def_return_dict
    else:
        def_return_dict["isSuccess"] = False
        return def_return_dict


@memberRouter.post("/register")
def register_member(
    email: str,
    password: str,
    member_name: str,
    store_name: str,
    cctv_url: str,
    cctv_name: str,
    session: Session = Depends(get_db),
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    temp = (
        session.query(models.Member)
        .filter(models.Member.email == email)
        .first()
    )
    if temp:
        def_return_dict["isSuccess"] = False
        return def_return_dict

    mem = models.Member(
        email=email,
        password=hash_password(password),
        member_name=member_name,
        store_name=store_name,
    )

    session.add(mem)
    session.commit()
    member = session.query(models.Member).filter_by(email=email).one()

    cctv = models.CCTV(
        cctv_url=cctv_url, cctv_name=cctv_name, member_id=member.member_id
    )
    session.add(cctv)
    session.commit()

    cctv.hls_url = (
        f"http://10.28.224.201:30576/hls/cctv_stream/{cctv.cctv_id}/index.m3u8"
    )
    session.commit()

    redis_server.lpush(
        "start_inf",
        json.dumps(
            {
                "cctv_id": cctv.cctv_id,
                "cctv_url": cctv.cctv_url,
                "threshold": member.threshold,
                "save_time_length": member.save_time_length,
            }
        ),
    )
    redis_server.lpush(
        "start_hls_stream",
        json.dumps(
            {
                "cctv_id": cctv.cctv_id,
                "cctv_url": cctv.cctv_url,
                "threshold": member.threshold,
                "save_time_length": member.save_time_length,
            }
        ),
    )
    return def_return_dict


@memberRouter.post("/duplicate")
def duplicate(email: str, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    mem = (
        session.query(models.Member)
        .filter(models.Member.email == email)
        .first()
    )
    if mem:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@memberRouter.post("/login")
def login(login_info: Login, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    try:
        member = (
            session.query(models.Member)
            .filter_by(email=login_info.email)
            .one()
        )
    except NoResultFound:
        def_return_dict["isSuccess"] = False
        return def_return_dict

    if verify_password(login_info.password, member.password):
        def_return_dict["result"] = {"member_id": member.member_id}
    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.post("/shop_edit")
def shop_edit(
    member_id: int, store_name: str, session: Session = Depends(get_db)
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    member = session.query(models.Member).get(member_id)
    if member:
        member.store_name = store_name
        session.commit()
    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.get("/profile_lookup")
def profile_lookup(member_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    member = session.query(models.Member).get(member_id)
    if member:
        def_return_dict["result"] = {
            "email": member.email,
            "password": member.password,
            "store_name": member.store_name,
        }
    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.post("/profile_edit")
def profile_edit(
    member_id: int,
    email: str,
    password: str,
    store_name: str,
    session: Session = Depends(get_db),
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    duplicate_email = (
        session.query(models.Member)
        .filter(models.Member.email == email)
        .first()
    )
    if duplicate_email:
        if duplicate_email.member_id != member_id:
            def_return_dict["isSuccess"] = False
            return def_return_dict

    member = session.query(models.Member).get(member_id)
    if member:
        member.email = email
        member.password = hash_password(password)
        member.store_name = store_name
        session.commit()

    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.post("/password_edit")
def password_edit(
    member_id: int, password: str, session: Session = Depends(get_db)
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    member = session.query(models.Member).get(member_id)
    if member:
        member.password = hash_password(password)
        session.commit()

    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.get("/alarm_lookup")
def alarm_lookup(member_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    member = session.query(models.Member).get(member_id)
    if member:
        def_return_dict["result"] = {
            "threshold": member.threshold,
            "save_time_length": member.save_time_length,
        }
    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.post("/alarm_edit")
def alarm_edit(
    member_id: int,
    threshold: float,
    save_time_length: int,
    session: Session = Depends(get_db),
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    member = session.query(models.Member).get(member_id)
    if member:
        member.threshold = threshold
        member.save_time_length = save_time_length
        session.commit()

        alarm_config = json.dumps(
            {"threshold": threshold, "save_time_length": save_time_length}
        )
        cctv_ids = session.query(models.CCTV.cctv_id).filter(
            models.Member.member_id == member_id,
            models.CCTV.cctv_delete_yn == False,  # noqa: E712
        )
        for cctv_id in cctv_ids:
            redis_server.lpush(f"{cctv_id[0]}_alarm", alarm_config)

    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.get("/cctv_list_lookup")
def cctv_list_lookup(member_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    cctvs = (
        session.query(models.CCTV)
        .filter_by(member_id=member_id)
        .order_by(models.CCTV.cctv_id.desc())
        .all()
    )
    data = [
        {
            "cctv_id": cctv.cctv_id,
            "cctv_name": cctv.cctv_name,
            "cctv_url": cctv.cctv_url,
            "hls_url": cctv.hls_url,
        }
        for cctv in cctvs
    ]
    if not data:
        def_return_dict["isSuccess"] = False

    else:
        def_return_dict["result"] = data

    return def_return_dict


@settingRouter.post("/cctv_register")
def cctv_register(
    member_id: int,
    cctv_name: str,
    cctv_url: str,
    session: Session = Depends(get_db),
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    member = session.query(models.Member).get(member_id)
    if member:
        cctv = models.CCTV(
            cctv_name=cctv_name, member_id=member_id, cctv_url=cctv_url
        )
        session.add(cctv)
        session.commit()

        cctv.hls_url = f"http://10.28.224.201:30576/hls/cctv_stream/{cctv.cctv_id}/index.m3u8"
        session.commit()

        def_return_dict["result"] = {
            "cctv_name": cctv.cctv_name,
            "cctv_id": cctv.cctv_id,
            "cctv_url": cctv_url,
        }
        redis_server.lpush(
            "start_inf",
            json.dumps(
                {
                    "cctv_id": cctv.cctv_id,
                    "cctv_url": cctv.cctv_url,
                    "threshold": member.threshold,
                    "save_time_length": member.save_time_length,
                }
            ),
        )
        redis_server.lpush(
            "start_hls_stream",
            json.dumps({"cctv_id": cctv.cctv_id, "cctv_url": cctv.cctv_url}),
        )
    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.delete("/cctv_delete")
def cctv_delete(cctv_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    cctv = session.query(models.CCTV).get(cctv_id)
    if not cctv:
        def_return_dict["isSuccess"] = False
    else:
        session.delete(cctv)
        session.commit()

        flag_key = f"{cctv_id}_stop_inf"
        redis_server.lpush(flag_key, "")
        redis_server.lpush("stop_hls_stream", cctv_id)

    return def_return_dict


@settingRouter.post("/cctv_edit")
def cctv_edit(
    cctv_id: int,
    cctv_name: str,
    cctv_url: str,
    session: Session = Depends(get_db),
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    cctv = session.query(models.CCTV).get(cctv_id)
    if not cctv:
        def_return_dict["isSuccess"] = False
    else:
        change_url = cctv.cctv_url != cctv_url
        cctv.cctv_name = cctv_name
        cctv.cctv_url = cctv_url
        session.commit()

        if change_url:
            threshold, save_time_length = (
                session.query(
                    models.Member.threshold, models.Member.save_time_length
                )
                .filter(models.Member.member_id == cctv.member_id)
                .first()
            )

            flag_key = f"{cctv_id}_stop_inf"
            redis_server.lpush(flag_key, "")
            redis_server.lpush(
                "start_inf",
                json.dumps(
                    {
                        "cctv_id": cctv.cctv_id,
                        "cctv_url": cctv.cctv_url,
                        "threshold": threshold,
                        "save_time_length": save_time_length,
                    }
                ),
            )
            redis_server.lpush("stop_hls_stream", cctv.cctv_id)
            redis_server.lpush(
                "start_hls_stream",
                json.dumps(
                    {"cctv_id": cctv.cctv_id, "cctv_url": cctv.cctv_url}
                ),
            )

    return def_return_dict


# ================ Hyunwoo ================
@cctvRouter.get("/loglist_lookup")
def select_loglist_lookup(member_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    try:
        tmp_list = []
        info_list = (
            session.query(models.Log, models.CCTV)
            .join(models.CCTV, models.CCTV.cctv_id == models.Log.cctv_id)
            .filter(
                models.CCTV.member_id == member_id,
                models.Log.anomaly_delete_yn == False,  # noqa: E712
            )
            .order_by(models.Log.log_id.desc())
            .all()
        )
        if info_list is None:
            raise Exception()

        for log_info, cctv_info in info_list:
            tmp_dict = dict()
            log_dict, cctv_dict = vars(log_info), vars(cctv_info)
            for k, v in log_dict.items():
                tmp_dict[k] = v
            for k, v in cctv_dict.items():
                tmp_dict[k] = v
            tmp_list.append(tmp_dict)
        def_return_dict["result"] = tmp_list
    except Exception:
        def_return_dict["isSuccess"] = False
    return def_return_dict


@cctvRouter.put("/feedback")
def update_log_feedback(
    log_id: int, feedback: int, session: Session = Depends(get_db)
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    try:
        log_info = (
            session.query(models.Log)
            .filter(models.Log.log_id == log_id)
            .first()
        )

        if log_info is None:
            raise Exception()

        log_info.anomaly_feedback = feedback
        session.commit()
    except Exception:
        def_return_dict["isSuccess"] = False
    return def_return_dict


@cctvRouter.post("/log_register")
def register_log(
    cctv_id: int,
    anomaly_create_time: str,
    anomaly_score: float,
    anomaly_save_path: str,
    video_file: UploadFile = File(...),
    session: Session = Depends(get_db),
):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    try:
        video_dir = os.path.dirname(video_file.filename)
        if not os.path.isdir(video_dir):
            os.makedirs(video_dir, exist_ok=True)

        with open(f"{video_file.filename}", "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        log = models.Log(
            cctv_id=cctv_id,
            anomaly_create_time=anomaly_create_time,
            anomaly_score=anomaly_score,
            anomaly_save_path=anomaly_save_path,
        )
        session.add(log)
        session.commit()

    except Exception:
        def_return_dict["isSuccess"] = False
    return def_return_dict


@cctvRouter.delete("/log_delete")
def delete_log(log_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    try:
        log_info = (
            session.query(models.Log)
            .filter(models.Log.log_id == log_id)
            .first()
        )

        if log_info is None:
            raise Exception()

        session.delete(log_info)
        session.commit()
    except Exception:
        def_return_dict["isSuccess"] = False
    return def_return_dict


@streamingRouter.get("/list_lookup")
def select_cctvlist(member_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    try:
        cctv_list = (
            session.query(models.CCTV)
            .filter(models.CCTV.member_id == member_id)
            .all()
        )

        if cctv_list is None:
            raise Exception()

        def_return_dict["result"] = cctv_list
    except Exception:
        def_return_dict["isSuccess"] = False
    return def_return_dict


@cctvRouter.get("/log_count")
def select_log_count(member_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    try:
        log_count = (
            session.query(models.Log.log_id)
            .join(models.CCTV, models.CCTV.cctv_id == models.Log.cctv_id)
            .join(
                models.Member, models.Member.member_id == models.CCTV.member_id
            )
            .filter(
                models.CCTV.member_id == member_id,
            )
            .count()
        )

        if log_count is None:
            raise Exception()

        def_return_dict["result"] = log_count
    except Exception:
        def_return_dict["isSuccess"] = False
    return def_return_dict


@cctvRouter.get("/{log_id}/video.mp4")
async def get_video(video_path: str, log_id: int):
    return FileResponse(video_path)


# API v 0.3.0
