from pydantic import Field, BaseSettings


class Config(BaseSettings):
    db_url: str = Field(
        default="postgresql://backend:password@10.28.224.142:30017/backend_db",
        env="DB_URL",
    )
    # db_url: str = Field(default="postgresql://backend:password@10.28.224.142:5433/backend_db",env='DB_URL')


config = Config()
