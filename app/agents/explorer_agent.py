"""
Explorer Agent

ReAct-based conversational agent for dynamic exploration.
Replaces rigid S5 state machine with intelligent reasoning loop.

Architecture:
    REASON (LLM) → ACT (Tools) → OBSERVE (Update) → Loop/Exit
"""

import os
from typing import Dict, Any, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agents.state import AgentState
from app.agents.memory import get_hybrid_memory
from app.agents.supervisor import AgentSupervisor
from app.agents.tools.rag_search import search_knowledge_base
from app.agents.tools.slot_writer import save_slot, get_slots, calculate_lead_score
from app.logger import get_logger

logger = get_logger("explorer_agent")

# System prompt for the Explorer Agent
EXPLORER_SYSTEM_PROMPT = """You are a helpful and consultative assistant for DITS (Digital IT Solutions).

Your role is to:
1. Understand the user's needs through natural conversation
2. Answer questions about services using the knowledge base
3. Gently gather information (name, challenges) when natural
4. Guide users toward specific solutions when they're ready

CRITICAL ANTI-HALLUCINATION RULES:
- NEVER claim to take actions you cannot take (e.g., "I've connected you", "I've marked you as ready", "You should receive details shortly")
- NEVER pretend to schedule calls, send emails, or connect users to real people
- NEVER say things like "I'll make sure to connect you", "Our team will reach out" unless this is ACTUALLY happening
- You are a chatbot that provides INFORMATION only - you cannot take real-world actions
- If user wants to talk to an expert or schedule a call, share the contact link: https://www.ditstek.com/contact-us
- For portfolios, case studies, or "See our work" requests (only when explicitly asked): https://www.ditstek.com/work or https://www.ditstek.com/blog
- Be HONEST about what you can and cannot do

ANTI-REPETITION RULES (CRITICAL):
- NEVER share the same link or resource twice in a conversation
- If you've already mentioned a service page or case study, reference it differently ("as I mentioned earlier" or "building on what we discussed")
- Vary your response structure - don't use the same opening or closing phrases
- If asked about the same topic again, go DEEPER instead of repeating the same overview
- Instead of re-sharing links, offer to discuss specific aspects: implementation details, timelines, pricing, case studies
- When you've exhausted information on a topic, acknowledge it and suggest moving forward: "I've shared the key resources on this. Would you like to discuss how this applies to your situation, or explore something else?"

Guidelines:
- **CRITICAL:** After searching the knowledge base ONCE, you MUST provide an answer based on the results. Do NOT search again for the same topic.
- Be conversational and friendly, not robotic
- Use the search_knowledge_base tool when users ask about services, pricing, or capabilities
- Use save_slot to remember important information the user shares
- If the user shares their name, save it with save_slot
- If the user describes their challenge/need, save it as context_signal
- **CRITICAL:** Proactively move the conversation forward. Do not wait for explicit "I am ready" commands.
- **Rule of Thumb:** If you have answered the user's question and they acknowledge it (in ANY way) without asking a new verification question, set is_ready=true.
- This applies to: expressions of thanks, agreement, understanding ("ok", "got it"), satisfaction, or statements of intent ("I want this").
- Default to moving the user to the next stage (CTAs) rather than lingering in the chat.
- Keep responses concise but helpful (2-3 sentences usually)
- Never fabricate information - use the knowledge base
- If search returns no results, honestly state that and ask for clarification instead of searching again.

Available Tools:
- search_knowledge_base: Search for company/service information
- save_slot: Save user information (user_name, context_signal, email)
- get_slots: Check what information is already collected
"""


class ExplorerAgent:
    """
    ReAct agent for dynamic conversation exploration.
    
    Integrates with existing orchestrator as a replacement for
    the rigid EXPLORATION state.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        supervisor: Optional[AgentSupervisor] = None
    ):
        """
        Initialize the Explorer Agent.
        
        Args:
            model: LLM model to use
            supervisor: Guardrails supervisor (uses default if None)
        """
        self.model_name = model
        self.supervisor = supervisor or AgentSupervisor()
        self.memory = get_hybrid_memory()
        
        # Initialize LLM with tools
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            streaming=True
        )
        
        # Bind tools to LLM
        self.tools = [search_knowledge_base, save_slot, get_slots]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"[ExplorerAgent] Initialized with model: {model}")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph ReAct graph.
        
        Nodes:
            - reason: LLM reasoning step
            - act: Tool execution
            - observe: State update
            
        Returns:
            Compiled StateGraph
        """
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("reason", self._reason_node)
        graph.add_node("act", ToolNode(self.tools))
        graph.add_node("observe", self._observe_node)
        
        # Entry point
        graph.set_entry_point("reason")
        
        # Edges
        graph.add_conditional_edges(
            "reason",
            self._should_use_tools,
            {
                "tools": "act",
                "respond": "observe"
            }
        )
        graph.add_edge("act", "observe")
        graph.add_conditional_edges(
            "observe",
            self._should_continue,
            {
                "continue": "reason",
                "ready": END,
                "exit": END
            }
        )
        
        return graph.compile()
    
    def _reason_node(self, state: AgentState) -> Dict[str, Any]:
        """
        LLM reasoning step - decides what to do next.
        
        CRITICAL: Loads conversation history from session memory to maintain context.
        """
        session_id = state.get("session_id", "")
        slots = state.get("slots", {})
        current_messages = state.get("messages", [])
        
        # Build messages array with FULL conversation history
        logger.info("<< PROMPT TRIGGER >> Using EXPLORER_SYSTEM_PROMPT")
        messages = [SystemMessage(content=EXPLORER_SYSTEM_PROMPT)]
        
        # Add slot context as system message (what we know about user)
        slot_context = []
        if slots.get("user_name"):
            slot_context.append(f"User's name: {slots['user_name']}")
        if slots.get("context_signal"):
            slot_context.append(f"User's need: {slots['context_signal']}")
        if slots.get("capability_bucket"):
            slot_context.append(f"Selected area: {slots['capability_bucket']}")
        
        if slot_context:
            context_msg = SystemMessage(
                content=f"IMPORTANT - Information already collected from this user:\n" + "\n".join(slot_context) + 
                "\n\nDO NOT ask for this information again. Reference it naturally."
            )
            messages.append(context_msg)
        
        # Load conversation history from session memory
        try:
            from app.utils.conversation_memory import get_session_memory_manager
            memory_mgr = get_session_memory_manager()
            memory = memory_mgr.get_or_create_memory(session_id)
            
            # Get recent messages (last 10 = 5 exchanges) for context
            history_messages = memory.chat_memory.messages[-10:] if memory.chat_memory.messages else []
            
            for msg in history_messages:
                # Handle dicts (from fallback memory) or objects (from LangChain memory)
                msg_role = None
                msg_content = None
                
                if isinstance(msg, dict):
                    msg_role = msg.get("role") or msg.get("type")
                    msg_content = msg.get("content")
                else:
                    msg_role = getattr(msg, "type", None) or getattr(msg, "role", None)
                    msg_content = getattr(msg, "content", None)

                if msg_role in ["human", "user"]:
                    messages.append(HumanMessage(content=msg_content))
                elif msg_role in ["ai", "assistant", "bot"]:
                    messages.append(AIMessage(content=msg_content))
            
            if history_messages:
                logger.debug(f"[ExplorerAgent] Loaded {len(history_messages)} history messages")
                
        except Exception as e:
            logger.warning(f"[ExplorerAgent] Could not load history: {e}")
        
        # Add current session messages (including tool calls/outputs)
        for msg in current_messages:
            if isinstance(msg, HumanMessage):
                # Check if this user message is already in history to avoid duplication
                is_duplicate = any(
                    isinstance(m, HumanMessage) and m.content == msg.content 
                    for m in messages
                )
                if not is_duplicate:
                    messages.append(msg)
            else:
                # Always include AI messages (tool calls) and Tool messages (results)
                messages.append(msg)
        
        # Call LLM with full context
        response = self.llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def _observe_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Update state based on latest response.
        
        KEY LOGIC: Track if a tool was just executed. Only continue
        the loop if tools were used; otherwise exit to return response.
        """
        messages = state.get("messages", [])
        turn_count = state.get("turn_count", 0) + 1
        
        # Check if last message was a tool result (meaning we should continue)
        tool_just_used = False
        if messages:
            last_msg = messages[-1]
            # Tool messages have a 'tool_call_id' or are ToolMessage type
            if hasattr(last_msg, 'tool_call_id') or (hasattr(last_msg, 'type') and last_msg.type == 'tool'):
                tool_just_used = True
        
        # Check for readiness/closing signals in USER's LAST input only
        is_ready = state.get("is_ready", False)
        
        # Find the last HumanMessage (most recent user input)
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and hasattr(msg, 'content') and msg.content:
                last_user_msg = msg.content.lower()
                break
        
        if last_user_msg:
            # CTA readiness signals - user wants to proceed
            if any(signal in last_user_msg for signal in [
                "ready", "i'm ready", "i am ready", "let's proceed",
                "show me options", "what's next", "next steps",
                "schedule", "book a call", "talk to", "speak to",
                "get a demo", "contact", "let's do it", "sign me up",
                "interested", "want to proceed", "move forward"
            ]):
                is_ready = True
                logger.info(f"[ExplorerAgent] User CTA readiness detected: '{last_user_msg[:50]}...'")
            
            # Closing signals - user is done exploring
            elif any(signal in last_user_msg for signal in [
                "thanks i'm good", "thanks i am good", "good for now",
                "that's all", "that is all", "i'm done", "im done",
                "bye", "goodbye", "no thanks", "not interested",
                "maybe later", "i'll pass", "thank you for", "thanks for your"
            ]):
                is_ready = True
                logger.info(f"[ExplorerAgent] User closing signal detected: '{last_user_msg[:50]}...'")
        
        # Calculate lead score
        lead_score = calculate_lead_score(
            state.get("session_id", ""),
            turn_count
        )
        
        return {
            "turn_count": turn_count,
            "is_ready": is_ready,
            "lead_score": lead_score,
            "tool_just_used": tool_just_used  # Track for continue decision
        }
    
    def _should_use_tools(self, state: AgentState) -> Literal["tools", "respond"]:
        """
        Check if LLM wants to use tools.
        """
        messages = state.get("messages", [])
        if not messages:
            return "respond"
        
        last_msg = messages[-1]
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools"
        
        return "respond"
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "ready", "exit"]:
        """
        Determine whether to continue the ReAct loop.
        
        CRITICAL: Only continue if a tool was just used (need to process result).
        If LLM gave a final response (no tools), EXIT immediately.
        This ensures ONE turn per user message.
        """
        # If user signaled readiness, show options
        if state.get("is_ready", False):
            return "ready"
        
        # If we just used a tool, continue to process the result
        if state.get("tool_just_used", False):
            # But still respect supervisor limits
            return self.supervisor.check_should_continue(state)
        
        # LLM gave a response without tools - EXIT (return to orchestrator)
        # This is the normal case: user asks, agent responds, done for this turn
        return "exit"
    
    def invoke(
        self,
        user_input: str,
        session_id: str,
        initial_slots: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input through the agent.
        
        Args:
            user_input: User's message
            session_id: Session identifier
            initial_slots: Pre-existing slot values
            
        Returns:
            Final agent state with response
        """
        # Set up RAG search context to filter already-shared content
        from app.agents.tools.rag_search import set_search_context
        shared_urls = initial_slots.get("shared_urls", []) if initial_slots else []
        set_search_context(session_id, shared_urls)
        
        # Build initial state
        state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "slots": initial_slots or {},
            "session_id": session_id,
            "turn_count": 0,
            "lead_score": 0,
            "is_ready": False,
            "context_summary": None,
            "tool_just_used": False
        }
        
        # Run graph
        result = self.graph.invoke(state)
        
        logger.info(
            f"[ExplorerAgent] Completed - turns: {result.get('turn_count')}, "
            f"ready: {result.get('is_ready')}, score: {result.get('lead_score')}"
        )
        
        return result
    
    def get_response_text(self, result: Dict[str, Any]) -> str:
        """
        Extract the final response text from agent result.
        
        Args:
            result: Agent invoke result
            
        Returns:
            Response text string (sanitized to remove internal IDs)
        """
        messages = result.get("messages", [])
        raw_text = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if not getattr(msg, 'tool_calls', None):
                    raw_text = msg.content
                    break
        
        if not raw_text:
            return "I'm here to help. What would you like to know?"
        
        # ============================================================
        # SANITIZE: Remove internal IDs and patterns from agent response
        # Same patterns as llm_adapter._sanitize_response()
        # ============================================================
        return self._sanitize_response(raw_text)
    
    def _sanitize_response(self, text: str) -> str:
        """
        Remove any leaked internal patterns from agent response.
        
        Catches cases where the model echoes internal IDs like UC1-A.
        """
        import re
        
        # Patterns that should NEVER appear in user-facing text
        forbidden_patterns = [
            r"\bUC1-[A-F]\b",  # Internal Bucket IDs (e.g. UC1-A, UC1-B)
            r"\bUC\d+-[A-Z0-9]+\b",  # Generic Internal IDs (UC1-A, UC2-X, etc.)
            r"UC-1-[A-F]",  # Variant format
            r"\bthe UC1?-?[A-F]? area\b",  # "the UC1-A area"
            r"in the UC1?-?[A-F]? ",  # "in the UC1-A category"
            r"Focus:.*?\|.*?Goal:.*?\|",  # Anchor echo
            r"\[Context\].*?\n\n",  # Context block header
        ]
        
        for pattern in forbidden_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up any double spaces or empty phrases left behind
        text = re.sub(r"  +", " ", text)
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        
        return text.strip()


# Singleton instance management
_agent_instance: Optional[ExplorerAgent] = None


def get_explorer_agent() -> ExplorerAgent:
    """Get or create singleton Explorer Agent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ExplorerAgent()
    return _agent_instance
