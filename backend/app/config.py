from pydantic import Field, BaseSettings


class Config(BaseSettings):
    db_url: str = Field(
        default="postgresql://backend:password@10.28.224.142:30017/backend_db",
        env="DB_URL",
    )
    # db_url: str = Field(default="postgresql://backend:password@10.28.224.142:5433/backend_db",env='DB_URL')
    smtp_server: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")

    smtp_ssl_port: int = Field(default=465, env="SMTP_SSL_PORT")

    sender_email: str = Field(
        default="gusdn00751@gmail.com", env="SENDER_EMAIL"
    )

    sender_pass: str = Field(default="wcrcavozgdpcehwd", env="SENDER_PASSWORD")

    redis_host: str = Field(default="10.28.224.201", env="REDIS_HOST")
    redis_port: int = Field(default=30435, env="REDIS_PORT")

    hls_url_template: str = Field(
        default="http://10.28.224.201:30438/hls/cctv_stream/{cctv_id}/index.m3u8",
        env="HLS_URL",
    )

    hls_root_dir: str = Field(
        default="/data/ephemeral/home/level2-3-cv-finalproject-cv-03/backend/hls/cctv_stream",
        env="HLS_ROOT_DIR",
    )


config = Config()
