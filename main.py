import os
import logging
import sys
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("chatbot")

# Configure file logging
log_file_path = os.path.join("logs", "backend.log")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Configure uvicorn logger
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.DEBUG)
for handler in logger.handlers:
    uvicorn_logger.addHandler(handler)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Initialize FastAPI app
    app = FastAPI(
        title="Chatbot API",
        description="API for the chatbot backend service",
        version="1.0.0"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: restrict in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize LLM
    from backend.llm_client import llm
    llm = llm
    logger.info("✅ LLM client initialized successfully")

    # Initialize FollowUpManager
    from backend.nested_follow_up_manager import FollowUpManager
    app.state.follow_up_manager = FollowUpManager(llm)
    logger.info("✅ FollowUpManager initialized successfully")

    # Include API routers
    from backend.api_v1 import router as api_v1_router
    app.include_router(api_v1_router)
    logger.info("✅ API v1 routes registered")

    return app

def main():
    """Main entry point for the application."""
    # Create FastAPI app
    app = create_app()
    logger.info("✅ Application initialized successfully")

    # Get configuration from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "False").lower() == "true"

    # Run the server
    uvicorn.run(
        "main:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
