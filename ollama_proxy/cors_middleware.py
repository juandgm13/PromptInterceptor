"""
CORS middleware for PyProxy.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config


def add_cors_middleware(app: FastAPI) -> None:
    """
    Add CORS middleware to FastAPI app.

    When allow_origins contains '*', allow_credentials must be False
    (browsers reject credentialed requests to wildcard origins per the
    CORS spec).
    """
    config = get_config()
    wildcard = "*" in config.allow_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_credentials=not wildcard,
        allow_methods=config.allow_methods,
        allow_headers=config.allow_headers,
        max_age=86400,
    )
