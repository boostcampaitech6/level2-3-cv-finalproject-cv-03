from fastapi import APIRouter, Depends
from app.schemas import Login
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db import models
from sqlalchemy.orm.exc import NoResultFound
from passlib.context import CryptContext

DEFAULT_RETURN_DICT = {"isSuccess": True, "result": None}


router = APIRouter()
memberRouter = APIRouter(prefix="/api/v0/members")
cctvRouter = APIRouter(prefix="/api/v0/cctv")
streamingRouter = APIRouter(prefix="/api/v0/streaming")
settingRouter = APIRouter(prefix="/api/v0/settings")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


@router.get("/")
def read_root():
    return {"Hello": "World!"}


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

    cctv = models.CCTV(
        cctv_url=cctv_url, cctv_name=cctv_name, member_id=mem.member_id
    )
    session.add(cctv)
    session.commit()

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

    else:
        def_return_dict["isSuccess"] = False

    return def_return_dict


@settingRouter.get("/cctv_list_lookup")
def cctv_list_lookup(member_id: int, session: Session = Depends(get_db)):
    def_return_dict = DEFAULT_RETURN_DICT.copy()

    cctvs = session.query(models.CCTV).filter_by(member_id=member_id).all()
    data = [
        {
            "cctv_id": cctv.cctv_id,
            "cctv_name": cctv.cctv_name,
            "cctv_url": cctv.cctv_url,
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
        def_return_dict["result"] = {
            "cctv_name": cctv.cctv_name,
            "cctv_id": cctv.cctv_id,
            "cctv_url": cctv_url,
        }
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
        cctv.cctv_name = cctv_name
        cctv.cctv_url = cctv_url
        session.commit()

    return def_return_dict


# ================ Hyunwoo ================


"""
GET /api/v0/cctv/loglist_lookup
GET /api/v0/cctv/log_lookup
POST /api/v0/cctv/feedback
DELETE /api/v0/cctv/log_delete
GET /api/v0/streaming/list_lookup
"""


@cctvRouter.get("/loglist_lookup")
def select_loglist_lookup(member_id: int, session: Session = Depends(get_db)):
    # ex) http://10.28.224.142:30016/api/v0/cctv/loglist_lookup?member_id=13
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


@cctvRouter.get("/log_lookup")
def select_log_lookup(log_id: int, session: Session = Depends(get_db)):
    # ex) http://10.28.224.142:30016/api/v0/cctv/log_lookup?log_id=1
    def_return_dict = DEFAULT_RETURN_DICT.copy()
    try:
        log_info = (
            session.query(models.Log)
            .filter(models.Log.log_id == log_id)
            .first()
        )

        if log_info is None:
            raise Exception()

        def_return_dict["result"] = log_info
    except Exception:
        def_return_dict["isSuccess"] = False
    return def_return_dict


@cctvRouter.put("/feedback")
def update_log_feedback(
    log_id: int, feedback: int, session: Session = Depends(get_db)
):
    """
    curl -X 'PUT' \
    'http://10.28.224.142:30016/api/v0/cctv/feedback?log_id=1&feedback=0' \
    -H 'accept: application/json'
    """
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


@cctvRouter.delete("/log_delete")
def delete_log(log_id: int, session: Session = Depends(get_db)):
    """
    curl -X 'DELETE' \
    'http://10.28.224.142:30016/api/v0/cctv/log_delete?log_id=3' \
    -H 'accept: application/json'
    """
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
    # ex) http://10.28.224.142:30016/api/v0/streaming/list_lookup?member_id=1
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


# API v 0.3.0
