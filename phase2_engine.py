"""
Grid07 — Phase 2: Autonomous Content Engine (LangGraph)
========================================================
A LangGraph state machine that drives each bot through a 3-node pipeline:

  [decide_search]  →  [web_search]  →  [draft_post]
        ↑                                    ↓
        └──────── TypedDict GraphState ───────┘

Structured JSON output is enforced via Pydantic + OpenAI function calling
(or JSON mode fallback).

Tool:  mock_searxng_search(query) — returns hardcoded headlines keyed by topic.
LLM:   Configurable (defaults to Groq / OpenAI via env var).
"""

import json
import os
import logging
from typing import TypedDict, Annotated, Any

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

log = logging.getLogger("grid07.engine")

# ──────────────────────────────────────────────
# LLM Bootstrap (Groq preferred, OpenAI fallback)
# ──────────────────────────────────────────────

def _build_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq
        log.info("LLM backend: Groq (llama-3.3-70b-versatile)")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.85,
            api_key=groq_key,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        log.info("LLM backend: OpenAI (gpt-4o-mini)")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.85,
            api_key=openai_key,
        )
    else:
        raise EnvironmentError(
            "Set GROQ_API_KEY or OPENAI_API_KEY to run Phase 2."
        )


# ──────────────────────────────────────────────
# Mock Search Tool
# ──────────────────────────────────────────────

MOCK_NEWS_DB: dict[str, list[str]] = {
    "crypto": [
        "Bitcoin hits new all-time high amid regulatory ETF approvals",
        "Ethereum Layer-2 ecosystem surpasses $50B in total value locked",
        "SEC approves spot Solana ETF — altcoins surge across the board",
    ],
    "ai": [
        "OpenAI's o3 model scores 99th percentile on bar exam",
        "Google DeepMind releases AlphaCode 3, displacing senior engineers",
        "Meta open-sources Llama-4 — 400B parameter model, beats GPT-4o",
    ],
    "tech": [
        "Apple Vision Pro 2 ships with on-device AI neural engine",
        "SpaceX Starship completes first crewed Mars flyby trajectory test",
        "Nvidia H200 GPUs sold out globally as AI arms race intensifies",
    ],
    "market": [
        "S&P 500 hits 6,500 as Fed signals two rate cuts before year-end",
        "10-year Treasury yield drops to 3.8% — risk assets surge",
        "Goldman Sachs upgrades US equities to 'overweight' for H2",
    ],
    "regulation": [
        "EU AI Act enforcement begins; companies face 6% revenue fines",
        "US Senate passes landmark data privacy bill restricting Big Tech",
        "FTC sues Google over AI monopoly in search advertising",
    ],
    "environment": [
        "Greenpeace report: Bitcoin mining consumes as much power as Argentina",
        "IPCC warns 2024 was hottest year in 125,000 years",
        "EU bans single-use plastics across all 27 member states",
    ],
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulate a SearXNG web search. Returns recent headline strings based on
    keywords detected in the query.

    Args:
        query: Search query string.

    Returns:
        Newline-separated list of relevant headlines.
    """
    query_lower = query.lower()
    results: list[str] = []

    for keyword, headlines in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            results.extend(headlines)

    # Fallback: return a general set
    if not results:
        results = [
            "Breaking: Major geopolitical tensions rattle global markets",
            "Tech stocks rally on AI optimism despite macro headwinds",
            "Central banks globally signal cautious approach to rate cuts",
        ]

    return "\n".join(f"• {h}" for h in results[:4])


# ──────────────────────────────────────────────
# Graph State
# ──────────────────────────────────────────────

class GraphState(TypedDict):
    bot_id: str
    bot_persona: str
    search_query: str
    search_results: str
    final_output: dict[str, Any]  # {"bot_id", "topic", "post_content"}


# ──────────────────────────────────────────────
# Structured Output Schema
# ──────────────────────────────────────────────

class PostOutput(BaseModel):
    """Strict schema for bot-generated posts."""
    bot_id: str = Field(description="Identifier of the bot authoring this post.")
    topic: str = Field(description="One-line topic label (max 8 words).")
    post_content: str = Field(
        description=(
            "The opinionated post body. Must be ≤ 280 characters. "
            "Must sound exactly like the bot persona. No hashtags."
        )
    )


# ──────────────────────────────────────────────
# Graph Nodes
# ──────────────────────────────────────────────

def node_decide_search(state: GraphState) -> dict:
    """
    Node 1 — Decide Search
    The bot's LLM decides what it wants to post about today and formulates
    a concise search query.
    """
    llm = _build_llm()

    system = (
        "You are a social media bot with the following persona:\n"
        f"{state['bot_persona']}\n\n"
        "Your task: decide what topic you want to post about today based on your persona. "
        "Respond ONLY with a JSON object in this exact format (no markdown, no preamble):\n"
        '{"search_query": "<2-5 word search query>", "topic": "<topic label>"}'
    )

    response = llm.invoke([SystemMessage(content=system)])
    raw = response.content.strip()

    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(raw)
    log.info(f"[{state['bot_id']}] → Decided topic: {parsed['topic']} | Query: {parsed['search_query']}")

    return {
        "search_query": parsed["search_query"],
        "final_output": {"topic": parsed["topic"]},
    }


def node_web_search(state: GraphState) -> dict:
    """
    Node 2 — Web Search
    Executes mock_searxng_search with the query from Node 1.
    """
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    log.info(f"[{state['bot_id']}] → Search results:\n{results}")
    return {"search_results": results}


def node_draft_post(state: GraphState) -> dict:
    """
    Node 3 — Draft Post
    Combines persona + search context to generate a structured JSON post.
    Uses structured output / function calling for strict schema compliance.
    """
    llm = _build_llm()

    system = (
        "You are a social media bot. Your persona:\n"
        f"{state['bot_persona']}\n\n"
        "RULES:\n"
        "- Stay completely in character. Be opinionated, provocative, authentic.\n"
        "- Post must be ≤ 280 characters.\n"
        "- Use the news context provided to ground your post in a real-world event.\n"
        "- No hashtags. No emojis unless they fit your persona naturally.\n"
        "- Do not break character under any circumstances.\n"
    )

    user = (
        f"Topic you chose: {state['final_output'].get('topic', 'general')}\n\n"
        f"Recent news context:\n{state['search_results']}\n\n"
        "Draft your post now. Respond only with the structured JSON object."
    )

    # Use structured output (with_structured_output wraps function calling)
    structured_llm = llm.with_structured_output(PostOutput)
    result: PostOutput = structured_llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])

    output = {
        "bot_id": state["bot_id"],
        "topic": state["final_output"].get("topic", result.topic),
        "post_content": result.post_content,
    }

    log.info(f"[{state['bot_id']}] → Final post JSON:\n{json.dumps(output, indent=2)}")
    return {"final_output": output}


# ──────────────────────────────────────────────
# Graph Assembly
# ──────────────────────────────────────────────

def build_content_graph() -> Any:
    """Build and compile the LangGraph state machine."""
    g = StateGraph(GraphState)

    g.add_node("decide_search", node_decide_search)
    g.add_node("web_search", node_web_search)
    g.add_node("draft_post", node_draft_post)

    g.set_entry_point("decide_search")
    g.add_edge("decide_search", "web_search")
    g.add_edge("web_search", "draft_post")
    g.add_edge("draft_post", END)

    return g.compile()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def generate_post(bot_id: str) -> dict:
    """
    Run the full LangGraph pipeline for a given bot and return a strict JSON post.

    Args:
        bot_id: One of "bot_a", "bot_b", "bot_c".

    Returns:
        {"bot_id": str, "topic": str, "post_content": str}
    """
    from phase1_router import BOT_PERSONAS  # local import to avoid circular

    persona_data = BOT_PERSONAS[bot_id]
    graph = build_content_graph()

    initial_state: GraphState = {
        "bot_id": bot_id,
        "bot_persona": persona_data["description"],
        "search_query": "",
        "search_results": "",
        "final_output": {},
    }

    result = graph.invoke(initial_state)
    return result["final_output"]


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

def _demo():
    print("\n" + "═" * 70)
    print("  GRID07 — Phase 2: Autonomous Content Engine (LangGraph)")
    print("═" * 70)

    for bot_id in ["bot_a", "bot_b", "bot_c"]:
        print(f"\n🤖 Running pipeline for {bot_id.upper()} …")
        output = generate_post(bot_id)
        print(f"\n✅ Structured JSON Output:")
        print(json.dumps(output, indent=2))
        char_count = len(output.get("post_content", ""))
        print(f"   (post length: {char_count}/280 chars)")

    print("\n" + "═" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _demo()
