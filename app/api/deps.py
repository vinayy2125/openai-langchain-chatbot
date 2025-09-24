from typing import Optional

from app.core.nested_follow_up_manager import FollowUpManager
from app.core.llm_client import llm


_follow_up_manager_instance: Optional[FollowUpManager] = None


def get_follow_up_manager() -> FollowUpManager:
    """Return a singleton FollowUpManager so in-memory session state persists across requests."""
    global _follow_up_manager_instance
    if _follow_up_manager_instance is None:
        _follow_up_manager_instance = FollowUpManager(llm=llm)
    return _follow_up_manager_instance
