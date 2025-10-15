from app.logger import get_logger
# Add project root to sys.path for module resolution
logger = get_logger(__name__)
logger.debug("App package initialized and sys.path updated.")
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
