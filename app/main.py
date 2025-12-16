"""Main entry point for the chatbot FastAPI application."""

import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.router import router as api_router
from fastapi.middleware.cors import CORSMiddleware

# from app.core.llm_client import llm
from app.api import redis_endpoint as redis_router

# from app.core.nested_follow_up_manager import FollowUpManager
from app.logger import get_logger
from app.logger import attach_handlers_to_uvicorn
from app.ingestion.scrape_to_redis import create_index_from_yaml
from pathlib import Path

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

    # Performance Monitoring Middleware
    from fastapi import Request
    import time
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Log slow requests (> 1 second)
        if process_time > 1.0:
             logger.warning(f"[Slow Request] {request.method} {request.url.path} took {process_time:.4f}s")
             
        return response

    logger.info("LLM client initialized successfully")

    # Initialize database connection pool
    try:
        from app.db.pool import initialize_pool
        initialize_pool()
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database connection pool: {e}")
        # Don't fail startup, pool will be initialized on first use

    # Attach FollowUpManager to app state - REMOVED for optimized flow only
    # app.state.follow_up_manager = FollowUpManager(llm)
    # logger.info("FollowUpManager initialized successfully")

    # Register main API router
    app.include_router(api_router)
    logger.info("API routes registered")
    # Register Redis endpoints router
    app.include_router(redis_router.router, prefix="/api")
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
    # Ensure chat_history_index exists
    try:
        yaml_path = Path(__file__).parent / "db" / "chat_history_index.yaml"
        create_index_from_yaml(str(yaml_path))
    except Exception:
        logger.exception("Failed to ensure chat_history_index on startup")

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
