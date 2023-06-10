from fastapi import APIRouter

from sapaback.api.v1.endpoint import tests
from sapaback.api.v1.endpoint import search

api_router = APIRouter()

api_router.include_router(tests.router, prefix="/tests", tags=["tests"])

api_router.include_router(search.router, prefix="/search", tags=["search"])
