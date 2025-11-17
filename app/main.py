"""Main entry point for the chatbot FastAPI application."""

import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# from app.core.llm_client import llm
from app.api.v1 import router as api_v1_router
from app.api.v1 import redis_endpoint as redis_router

# from app.core.nested_follow_up_manager import FollowUpManager
from app.logger import get_logger
from app.logger import attach_handlers_to_uvicorn

load_dotenv()


logger = get_logger("__main__")


# Factory function to create the FastAPI app
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Initialize FastAPI app
    app = FastAPI(
        title="Chatbot API",
        description="API for the chatbot backend service",
        version="1.0.0",
    )

    # Configure CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: restrict in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("LLM client initialized successfully")

    # Attach FollowUpManager to app state - REMOVED for optimized flow only
    # app.state.follow_up_manager = FollowUpManager(llm)
    # logger.info("FollowUpManager initialized successfully")

    # Register main API router
    app.include_router(api_v1_router)
    logger.info("API v1 routes registered")
    # Register Redis endpoints router
    app.include_router(redis_router.router, prefix="/api/v1")
    logger.info("Redis endpoints registered")

    return app


# Main function to start the server
def main():
    """Main entry point for the application."""
    # Create FastAPI app
    app = create_app()
    logger.info("Application initialized successfully")

    # Get configuration from environment variables
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "False").lower() == "true"

    # Attach our shared handlers to Uvicorn so server logs use the same handlers
    try:
        attach_handlers_to_uvicorn()
    except Exception:
        logger.exception(
            "Failed to attach handlers to uvicorn; continuing with default logging configuration"
        )

    # Run the Uvicorn server
    uvicorn.run(
        "main:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
