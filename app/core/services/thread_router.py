import logging
from typing import  Dict, Optional

logger = logging.getLogger(__name__)

class ThreadManager:
    def __init__(self):
        self.threads: Dict[str, Dict] = {}
        self.active_thread_id: Optional[str] = None


class ChatRouter:
    def __init__(self, followup_manager):
        self.followup_manager = followup_manager
        self.manager = ThreadManager()
