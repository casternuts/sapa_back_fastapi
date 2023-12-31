import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, HttpUrl, PostgresDsn, validator
import os
from pathlib import Path


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    SERVER_NAME: str = "sapa"
    SERVER_HOST: AnyHttpUrl = "http://localhost"
    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:3000"]

    #유료키(스타트업데이)
    # OPEN_API_KEY: str = os.environ['OPEN_API_KEY']
    # naver_client_id: str = os.environ['naver_client_id']
    # naver_client_secret: str = os.environ['naver_client_secret']
    # google_search_engine_key:str = os.environ['google_search_engine_key']
    # google_search_api_key:str =os.environ['google_search_api_key']

    OPEN_API_KEY: str
    naver_client_id: str
    naver_client_secret: str
    google_search_engine_key: str
    google_search_api_key: str


    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    PROJECT_NAME: str = "sapa_back"
    SENTRY_DSN: Optional[HttpUrl] = None


    class Config:
        env_file = f"{Path(__file__).resolve().parent}/.env"
        env_file_encoding = 'utf-8'



settings = Settings()
