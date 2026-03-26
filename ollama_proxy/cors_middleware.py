"""
CORS middleware for PyProxy.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config


def add_cors_middleware(app: FastAPI) -> None:
    """
    Add CORS middleware to FastAPI app.

    Args:
        app: FastAPI app instance
    """
    config = get_config()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_credentials=True,
        allow_methods=config.allow_methods,
        allow_headers=config.allow_headers,
        max_age=86400,
    )


def setup_cors_config(app: FastAPI) -> None:
    """
    Setup CORS configuration for FastAPI app.

    Args:
        app: FastAPI app instance
    """
    config = get_config()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_credentials=config.allow_origins and "*" in config.allow_origins,
        allow_methods=config.allow_methods,
        allow_headers=config.allow_headers,
        max_age=86400,
    )

    # Include CORS headers in all responses
    @app.middleware("http")
    async def add_cors_headers(request: Request, call_next):
        response = await call_next(request)

        # Add CORS headers
        for header in ["origin", "access-control-allow-origin", "access-control-allow-credentials",
                       "access-control-allow-methods", "access-control-allow-headers"]:
            response.headers[header] = ", ".join(config.allow_methods) if header == "access-control-allow-methods" else \
                ", ".join(config.allow_headers) if header == "access-control-allow-headers" else \
                ", ".join(config.allow_origins) if header == "access-control-allow-origin" else \
                "true" if header == "access-control-allow-credentials" else \
                request.headers.get(header, "")

        return response
