"""
API Security, authentication, and performance timing middleware.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from fastapi import HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

API_KEY_NAME = "X-API-Key"
DEFAULT_API_KEY = os.getenv("API_KEY", "sc-tower-secret-key-2026")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the API key passed in request header.
    Allow bypassing if API_KEY is set to 'disabled' or in local dev mode when no key is configured.
    """
    allowed_key = os.getenv("API_KEY", DEFAULT_API_KEY)
    if allowed_key.lower() == "disabled":
        return "disabled"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header. Provide a valid authentication key.",
        )

    if api_key != allowed_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid X-API-Key header credentials.",
        )

    return api_key


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Adds process execution time in milliseconds to response headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        return response
