import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

class ThreadManager:
    def __init__(self):
        self.threads: Dict[str, Dict] = {}
        self.active_thread_id: Optional[str] = None

    def start_thread(self, topic: str, first_message: str) -> str:
        """Create a new thread with unique id"""
        thread_id = f"thread_{len(self.threads)+1}_{int(datetime.now().timestamp())}"
        self.threads[thread_id] = {
            "topic": topic,
            "messages": [{"role": "user", "content": first_message}],
            "created_at": datetime.now(),
        }
        self.active_thread_id = thread_id
        return thread_id

    def add_message(self, thread_id: str, role: str, content: str):
        if thread_id not in self.threads:
            return
        self.threads[thread_id]["messages"].append({"role": role, "content": content})

    def find_relevant_thread(self, query: str, similarity_fn: Callable[[str, str], float], threshold: float = 0.65) -> Optional[str]:
        """Return thread_id of most relevant past thread"""
        best_match, best_score = None, 0.0
        for tid, data in self.threads.items():
            score = similarity_fn(query, data["topic"])
            if score > best_score:
                best_match, best_score = tid, score
        if best_score >= threshold:
            return best_match
        return None


def detect_company_intent(query: str) -> bool:
    """Simple keyword-based intent check (replace with classifier if needed)"""
    company_keywords = ["company", "about you", "where located", "who are you", "your services", "what do you do"]
    return any(kw in query.lower() for kw in company_keywords)


def handle_company_query(query: str) -> List[str]:
    """Direct factual answer with suggestions"""
    answer = "We are a software solutions company specializing in AI-powered chatbots and automation."
    suggestions = [
        "- Learn more about our services",
        "- Return to your ongoing project discussion",
    ]
    return [answer] + suggestions


class ChatRouter:
    def __init__(self, followup_manager):
        self.followup_manager = followup_manager
        self.manager = ThreadManager()

    def handle_user_query(self, user_query: str, prompt_context: str):
        # 1. Company intent override
        if detect_company_intent(user_query):
            return handle_company_query(user_query)

        # 2. Check existing threads
        thread_id = self.manager.find_relevant_thread(user_query)
        if thread_id:
            self.manager.active_thread_id = thread_id
            history = self.manager.threads[thread_id]["messages"]
            return self.followup_manager.stream_follow_up_generation(
                conversation_history=history,
                latest_query=user_query,
                prompt_context=prompt_context,
                combined=True,
                followup_count=2
            )

        # 3. New thread
        new_tid = self.manager.start_thread(topic=user_query, first_message=user_query)
        return self.followup_manager.stream_follow_up_generation(
            conversation_history=self.manager.threads[new_tid]["messages"],
            latest_query=user_query,
            prompt_context=prompt_context,
            combined=False
        )
