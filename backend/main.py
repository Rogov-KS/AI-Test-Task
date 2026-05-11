from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import router
from backend.core import configure_logging, get_logger, settings

logger = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    logger.info("application initialized")
    return app


app = create_app()
