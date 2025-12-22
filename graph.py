import os
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from supabase_client import get_supabase

# Import Agents
from agents import (
    CoreIdentityArchitect,
    PurposeMotivationNavigator,
    GrandStrategyDirector,
    CapabilityGrowthEngineer,
    WorkplaceDynamicsCultureCoach,
    ChiefMarketingOfficer,
    QueryParser,
    ResponseSynthesizer,
    WebSearcher,
    ProfileUpdater,
)

# Load environment variables
load_dotenv()

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_profile: Dict[str, Any]  # Shared memory for structured user data
    active_agents: List[str]      # List of agents selected by the router
    agent_outputs: Dict[str, str] # Outputs from the specialist agents for the synthesizer
    web_search_results: str | None  # Optional shared web search context

# Initialize LLM (Google Gemini)
# Use a currently supported chat model; see Google AI docs for options.
# max_tokens caps response length to control cost across all agents.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=512,
)

# A lighter-outputs LLM variant for utility-style agents where
# short, factual responses are sufficient (router, web search,
# profile updates, history summarization).
utility_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=256,
)

# Initialize Agent Classes
identity_agent = CoreIdentityArchitect(llm)
purpose_agent = PurposeMotivationNavigator(llm)
strategy_agent = GrandStrategyDirector(llm)
capability_agent = CapabilityGrowthEngineer(llm)
dynamics_agent = WorkplaceDynamicsCultureCoach(llm)
cmo_agent = ChiefMarketingOfficer(llm)

# Utility agents use the smaller-output LLM to minimize cost where
# only compact facts or structured updates are needed.
router = QueryParser(utility_llm)
synthesizer = ResponseSynthesizer(llm)
web_searcher = WebSearcher(utility_llm)
profile_updater = ProfileUpdater(utility_llm)

# --- Node Functions ---

def router_node(state: AgentState):
    """Analyzes the user query and selects the appropriate agents."""
    last_message = state["messages"][-1].content
    result = router.get_chain().invoke({"input": last_message})

    # Base list from the router LLM
    destination_agents = list(getattr(result, "destination_agents", []) or [])

    # --- 1) Limit how many specialist agents run per query ---
    max_specialist_agents = 3
    specialist_order = [
        "core_identity_architect",
        "purpose_motivation_navigator",
        "grand_strategy_director",
        "capability_growth_engineer",
        "workplace_dynamics_coach",
        "chief_marketing_officer",
    ]

    # Preserve the router's ordering but cap the number of specialists.
    selected_specialists: List[str] = []
    for agent_id in destination_agents:
        if agent_id == "web_searcher":
            continue
        if agent_id in specialist_order and agent_id not in selected_specialists:
            selected_specialists.append(agent_id)
    selected_specialists = selected_specialists[:max_specialist_agents]

    # --- 2) Make web search more selective ---
    lowered = last_message.lower()
    web_keywords = [
        "search",
        "google",
        "news",
        "latest",
        "today",
        "current",
        "trend",
        "salary",
        "salaries",
        "market",
        "report",
        "reports",
        "statistic",
        "statistics",
        "stats",
        "2023",
        "2024",
        "2025",
        "data",
        "industry",
        "research",
        "article",
        "articles",
        "online",
        "website",
        "websites",
    ]
    needs_web_search = any(k in lowered for k in web_keywords)
    wants_web_search = "web_searcher" in destination_agents

    active_agents: List[str] = []
    active_agents.extend(selected_specialists)

    # Only keep the web_searcher when the query clearly needs fresh data,
    # or when the router chose ONLY the web_searcher and no specialists.
    if wants_web_search:
        if needs_web_search or not selected_specialists:
            active_agents.append("web_searcher")

    return {"active_agents": active_agents}


def web_search_node(state: AgentState):
    """Runs the WebSearcher agent (if selected) and stores shared web context."""
    last_message = state["messages"][-1].content

    # Build a short recent history for the web searcher to avoid huge prompts
    full_history = state.get("messages", [])
    history = full_history[-6:]

    # Use the WebSearcher helper to run Serper + LLM summarization without
    # relying on any model-specific tool-calling APIs.
    content = web_searcher.run(last_message, history)

    # Store as global web context and also as an agent output under a fixed key
    new_agent_outputs = dict(state.get("agent_outputs", {}))
    new_agent_outputs["web_searcher"] = content

    return {
        "web_search_results": content,
        "agent_outputs": new_agent_outputs,
    }

def run_agent(
    agent_instance,
    state: AgentState,
    agent_name: str,
    prior_agent_insights: str = "",
):
    """Helper to run a specialist agent.

    In addition to the raw user message, we inject:
    - Shared user_profile data.
    - Shared web_search_results (if any).
    - Optionally, summarized outputs from other agents that have already
      run in this turn (prior_agent_insights).
    """

    last_message = state["messages"][-1].content

    # Compact user profile context (avoid sending an unbounded dict string).
    profile = state.get("user_profile", {})
    profile_str = str(profile)
    if len(profile_str) > 1200:
        profile_str = profile_str[:1200] + "... (truncated)"
    profile_context = f"\n\n[Shared User Profile Data]: {profile_str}"

    # Compact shared web search context if present.
    web_context = ""
    if state.get("web_search_results"):
        web_text = str(state["web_search_results"])
        if len(web_text) > 1200:
            web_text = web_text[:1200] + "... (truncated)"
        web_context = f"\n\n[Shared Web Search Data]: {web_text}"

    # Limit how much of the prior agents' insights we resend.
    insights_context = ""
    if prior_agent_insights:
        insights_text = str(prior_agent_insights)
        if len(insights_text) > 2000:
            # Keep the most recent portion of the insights.
            insights_text = insights_text[-2000:]
        insights_context = (
            "\n\n[Other Specialist Agents' Insights So Far]:\n"
            f"{insights_text}"
        )

    full_input = last_message + profile_context + web_context + insights_context

    # Only send a short recent history window to each specialist.
    history = state.get("messages", [])
    short_history = history[-6:]

    response = agent_instance.get_chain().invoke({
        "input": full_input,
        "history": short_history,
    })
    content = getattr(response, "content", str(response))
    return {agent_name: content}

# Individual Agent Nodes
def specialist_agents_node(state: AgentState):
    """Runs all selected specialist agents (except web_searcher) and aggregates outputs."""
    outputs: Dict[str, str] = dict(state.get("agent_outputs", {}))
    active = state.get("active_agents", [])

    # Accumulate a simple text summary of previous agents' outputs so that
    # later agents in this turn can see what has already been concluded.
    prior_insights_str = ""

    for agent_id in active:
        if agent_id == "web_searcher":
            # Already handled (if selected) by web_search_node
            continue

        if agent_id == "core_identity_architect":
            result = run_agent(
                identity_agent,
                state,
                "Core Identity Architect",
                prior_agent_insights=prior_insights_str,
            )
        elif agent_id == "purpose_motivation_navigator":
            result = run_agent(
                purpose_agent,
                state,
                "Purpose Navigator",
                prior_agent_insights=prior_insights_str,
            )
        elif agent_id == "grand_strategy_director":
            result = run_agent(
                strategy_agent,
                state,
                "Strategy Director",
                prior_agent_insights=prior_insights_str,
            )
        elif agent_id == "capability_growth_engineer":
            result = run_agent(
                capability_agent,
                state,
                "Capability Engineer",
                prior_agent_insights=prior_insights_str,
            )
        elif agent_id == "workplace_dynamics_coach":
            result = run_agent(
                dynamics_agent,
                state,
                "Dynamics Coach",
                prior_agent_insights=prior_insights_str,
            )
        elif agent_id == "chief_marketing_officer":
            result = run_agent(
                cmo_agent,
                state,
                "Chief Marketing Officer",
                prior_agent_insights=prior_insights_str,
            )
        else:
            # Unknown agent id; skip
            continue

        # Merge this agent's output into the aggregated outputs
        outputs.update(result)

        # Also append it into the shared insights string for later agents
        for name, text in result.items():
            prior_insights_str += f"--- {name} ---\n{text}\n\n"

    return {"agent_outputs": outputs}

def synthesizer_node(state: AgentState):
    """Synthesizes all agent outputs into a final response."""
    user_query = state["messages"][-1].content
    agent_outputs = state["agent_outputs"]
    
    # Format outputs for the synthesizer, but cap total length to keep context small
    formatted_outputs = "\n\n".join([f"--- {k} ---\n{v}" for k, v in agent_outputs.items()])
    if len(formatted_outputs) > 4000:
        formatted_outputs = formatted_outputs[:4000] + "... (truncated)"
    
    response = synthesizer.get_chain().invoke({
        "user_query": user_query,
        "agent_outputs": formatted_outputs
    })
    
    return {"messages": [AIMessage(content=response.content)]}


def profile_updater_node(state: AgentState):
    """Updates the long-term user_profile based on recent conversation."""
    # Run this less frequently to save tokens: only on every 3rd user turn.
    messages = state.get("messages", [])
    human_count = sum(1 for m in messages if getattr(m, "type", None) == "human")
    if human_count % 3 != 0:
        return {}

    # Build a compact text representation of recent messages
    recent_messages = state.get("messages", [])[-8:]
    conversation_lines = []
    for msg in recent_messages:
        role = msg.type if hasattr(msg, "type") else msg.__class__.__name__
        content = getattr(msg, "content", str(msg))
        conversation_lines.append(f"[{role}] {content}")

    conversation_text = "\n".join(conversation_lines)

    current_profile = state.get("user_profile", {})

    result = profile_updater.get_chain().invoke(
        {
            "current_profile": current_profile,
            "conversation_text": conversation_text,
        }
    )

    updated = dict(current_profile)
    # Merge only the new/changed fields
    for key, value in (result.updated_profile or {}).items():
        updated[key] = value

    return {"user_profile": updated}


def history_manager_node(state: AgentState):
    """Trim and summarize long conversation history to stay within context window.

    If the number of messages is small, do nothing.
    If it grows large, summarize older messages into a single AIMessage and
    keep only the most recent messages verbatim. This keeps context rich
    while controlling token usage and cost.
    """

    messages = state.get("messages", [])
    # Raise the threshold so summarization happens less often.
    max_messages = 60
    keep_recent = 10

    if len(messages) <= max_messages:
        return {}

    older = messages[:-keep_recent]
    # Only summarize a bounded window of older messages to avoid huge prompts.
    max_older_messages = 40
    if len(older) > max_older_messages:
        older = older[-max_older_messages:]
    recent = messages[-keep_recent:]

    lines = []
    for msg in older:
        role = msg.type if hasattr(msg, "type") else msg.__class__.__name__
        content = getattr(msg, "content", str(msg))
        lines.append(f"[{role}] {content}")

    older_text = "\n".join(lines)

    summary_prompt = (
        "Summarize the following earlier part of the conversation in a very "
        "concise way, focusing only on stable preferences, goals, constraints, "
        "and key decisions.\n\n" f"Conversation so far:\n{older_text}"
    )

    # Use the smaller-output LLM here; the summary only needs to be
    # short and factual.
    summary_response = utility_llm.invoke(summary_prompt)
    summary_content = getattr(summary_response, "content", str(summary_response))

    summary_message = AIMessage(
        content=f"(Summary of earlier conversation)\n{summary_content}"
    )

    new_messages = [summary_message] + recent

    return {"messages": new_messages}


# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("web_searcher", web_search_node)
workflow.add_node("specialist_agents", specialist_agents_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("profile_updater", profile_updater_node)
workflow.add_node("history_manager", history_manager_node)

# Set Entry Point
workflow.set_entry_point("router")


def router_next(state: AgentState) -> str:
    """Decide whether to run web_searcher first or go straight to specialists."""
    active = state.get("active_agents", [])
    if "web_searcher" in active:
        return "web_searcher"
    return "specialist_agents"


# From router, either go to web_searcher (if selected) or straight to specialists
workflow.add_conditional_edges(
    "router",
    router_next,
    {
        "web_searcher": "web_searcher",
        "specialist_agents": "specialist_agents",
    },
)

# If web_searcher runs, always continue to specialists
workflow.add_edge("web_searcher", "specialist_agents")

# From specialists to synthesizer, then profile updater, then history manager, then end
workflow.add_edge("specialist_agents", "synthesizer")
workflow.add_edge("synthesizer", "profile_updater")
workflow.add_edge("profile_updater", "history_manager")
workflow.add_edge("history_manager", END)

# Compile
app = workflow.compile()

def _db_role_from_message(msg: Any) -> str:
    """Map a LangChain message to a DB role string."""

    role = getattr(msg, "type", None)
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    if role == "system":
        return "system"
    return "assistant"


def _message_from_db_row(row: Dict[str, Any]) -> Any:
    """Create a LangChain message from a DB row with role/content."""

    role = row.get("role", "assistant")
    content = row.get("content", "")
    if role == "user":
        return HumanMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    # Treat both assistant/summary/other as AI messages
    return AIMessage(content=content)


def load_user_profile(user_id: str) -> Dict[str, Any]:
    """Load (or initialize) the long-term user_profile from Supabase."""

    sb = get_supabase()
    resp = (
        sb.table("profiles")
        .select("data")
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )

    data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
    if data:
        # modern supabase-py returns either a dict or a list of dicts
        row = data if isinstance(data, dict) else data[0]
        return row.get("data", {}) or {}

    # If no profile exists yet (or we cannot see one due to RLS),
    # ensure an empty row exists using an upsert to avoid duplicate-key errors.
    sb.table("profiles").upsert({"user_id": user_id, "data": {}}).execute()
    return {}


def save_user_profile(user_id: str, profile: Dict[str, Any]) -> None:
    """Persist the user_profile back to Supabase."""

    sb = get_supabase()
    sb.table("profiles").upsert({"user_id": user_id, "data": profile}).execute()


def get_or_create_session(user_id: str, session_id: str | None, title: str | None) -> str:
    """Return a valid session_id for this user, creating a new row if needed.

    If session_id is provided, it is returned as-is (assumed valid). If not,
    a new chat_sessions row is created using the provided title.
    """

    sb = get_supabase()
    if session_id:
        return session_id

    if not title:
        title = "New session"

    # In supabase-py v2, insert() returns the inserted rows by default
    # when returning="representation" (the default). There is no .select()
    # method on the insert builder, so we just execute and read the data.
    resp = sb.table("chat_sessions").insert({"user_id": user_id, "title": title}).execute()

    data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
    if isinstance(data, dict):
        return data.get("id")
    if isinstance(data, list) and data:
        return data[0].get("id")

    raise RuntimeError("Failed to create or retrieve chat session ID from Supabase.")


def load_session_messages(session_id: str) -> List[Any]:
    """Load all messages for a given session from Supabase, oldest first."""

    sb = get_supabase()
    resp = (
        sb.table("messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )

    data = getattr(resp, "data", None) or resp.get("data") if isinstance(resp, dict) else None
    rows = data or []
    return [_message_from_db_row(row) for row in rows]


def append_session_messages(session_id: str, messages: List[Any]) -> None:
    """Append new messages for this session into Supabase."""

    if not messages:
        return

    sb = get_supabase()
    rows = []
    for msg in messages:
        content = getattr(msg, "content", str(msg))
        role = _db_role_from_message(msg)
        rows.append({
            "session_id": session_id,
            "role": role,
            "content": content,
        })

    sb.table("messages").insert(rows).execute()


def list_user_sessions(user_id: str) -> List[Dict[str, Any]]:
    """Return a list of this user's chat sessions (for sidebar-style UI)."""

    sb = get_supabase()
    resp = (
        sb.table("chat_sessions")
        .select("id, title, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    data = getattr(resp, "data", None) or resp.get("data") if isinstance(resp, dict) else None
    return data or []


def get_session_messages(session_id: str) -> List[Dict[str, str]]:
    """Return messages for a session as simple role/content dicts for UIs.

    Roles are normalized to "user", "assistant", or "system" to match
    common chat UI expectations.
    """

    lc_messages = load_session_messages(session_id)
    normalized: List[Dict[str, str]] = []

    for msg in lc_messages:
        role = getattr(msg, "type", None)
        if role == "human":
            ui_role = "user"
        elif role == "ai":
            ui_role = "assistant"
        elif role == "system":
            ui_role = "system"
        else:
            ui_role = "assistant"

        content = getattr(msg, "content", str(msg))
        normalized.append({"role": ui_role, "content": content})

    return normalized


def run_session(
    user_id: str,
    user_input: str,
    session_id: str | None = None,
) -> Dict[str, Any]:
    """High-level helper: run one turn of a chat session with persistence.

    - Loads user_profile and previous messages from Supabase.
    - Runs the LangGraph app for the new user_input.
    - Saves updated profile and new messages back to Supabase.
    - Returns the session_id and the assistant's latest reply.
    """

    # 1) Load long-term profile
    profile = load_user_profile(user_id)

    # 2) Get or create session row
    title = (user_input[:60] + "...") if user_input and len(user_input) > 60 else user_input
    session_id = get_or_create_session(user_id, session_id, title)

    # 3) Load previous messages
    past_messages = load_session_messages(session_id)

    # 4) Build initial state for this turn
    initial_messages = past_messages + [HumanMessage(content=user_input)]
    initial_state: AgentState = {
        "messages": initial_messages,
        "user_profile": profile,
        "active_agents": [],
        "agent_outputs": {},
        "web_search_results": None,
    }

    # 5) Run the graph synchronously to get the final state
    final_state = app.invoke(initial_state)

    final_messages = final_state["messages"]
    updated_profile = final_state.get("user_profile", {})

    # 6) Persist profile and only the new messages
    save_user_profile(user_id, updated_profile)

    new_messages = final_messages[len(past_messages) :]
    append_session_messages(session_id, new_messages)

    # 7) Extract the latest assistant reply for convenience
    latest_reply = ""
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage):
            latest_reply = msg.content
            break

    return {
        "session_id": session_id,
        "reply": latest_reply,
        "profile": updated_profile,
    }


# Example usage (for manual testing only):
if __name__ == "__main__":
    print("Running a sample persistent session turn...")
    try:
        result = run_session(
            user_id="demo-user",
            user_input=(
                "I want to move from marketing into data science, "
                "but I'm worried my math skills are too weak."
            ),
        )
        print("Session ID:", result["session_id"])
        print("Assistant Reply:\n", result["reply"])
    except Exception as e:
        print("Error running persistent session:", e)
        print(
            "Make sure GOOGLE_API_KEY, SUPABASE_URL, and SUPABASE_ANON_KEY "
            "are set, and the Supabase tables (profiles, chat_sessions, messages) exist."
        )
