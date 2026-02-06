"""
Agent Tools Package

Contains tools available to the Explorer Agent for:
- RAG search
- Slot management
- Lead scoring
"""

from app.agents.tools.rag_search import search_knowledge_base
from app.agents.tools.slot_writer import save_slot, get_slots

__all__ = ["search_knowledge_base", "save_slot", "get_slots"]
