from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.db.database import Base, engine
from app.api import (
    router,
    memberRouter,
    cctvRouter,
    streamingRouter,
    settingRouter,
)
from app.db.models import *
from app.middleware import TimeHeaderMiddleware
from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 데이터베이스 테이블 생성
    # logging.basicConfig()
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    Base.metadata.create_all(bind=engine)
    yield

def get_application():
    _app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

    _app.add_middleware(
        CORSMiddleware,
        # allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.include_router(router)
    _app.include_router(memberRouter)
    _app.include_router(cctvRouter)
    _app.include_router(streamingRouter)
    _app.include_router(settingRouter)
    _app.add_middleware(TimeHeaderMiddleware)

    _app.mount("/hls", StaticFiles(directory="hls"), name="hls")

    return _app


app = get_application()
