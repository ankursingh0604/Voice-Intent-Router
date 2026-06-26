import os
import json
from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────────

class RouterState(TypedDict):
    # Input
    transcript: str               # what Whisper transcribed
    transcription_confidence: Optional[str]  # rough self-check, see transcribe.py

    # Classification
    category: Optional[str]       # one of the 6 fixed categories
    classification_reasoning: Optional[str]

    # Routing results
    search_results: Optional[list]
    response: Optional[str]
    response_type: Optional[str]  # "text" | "code" — drives how UI renders it
    sources: Optional[list]

# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm(temperature=0.2):
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=temperature
    )

# ── Fixed category set ───────────────────────────────────────────────────────
# IMPORTANT: this list must exactly match the keys used in
# build_router_agent()'s add_conditional_edges() mapping below. LangGraph
# routes on the literal string returned by classify_input — if the LLM
# returns something outside this set, the graph would have no edge to
# follow. That's why classify_input() below forces the LLM's raw output
# through a strict validation step that falls back to "unclear_or_other"
# for ANYTHING that isn't an exact match, rather than trusting the LLM's
# string output directly.
VALID_CATEGORIES = [
    "factual_question",
    "time_sensitive_question",
    "opinion_question",
    "coding_request",
    "small_talk",
    "unclear_or_other",
]

# ── Node 1 — Classify Input ───────────────────────────────────────────────────

def classify_input(state: RouterState) -> dict:
    """
    Classify the transcribed input into one of VALID_CATEGORIES.

    LOOPHOLE GUARDED HERE: LLM classifiers don't always return exactly the
    string you ask for — they might add punctuation, wrap it in quotes, use
    different casing, or occasionally hallucinate a category that doesn't
    exist. Since LangGraph's conditional routing needs an EXACT string
    match against the edges defined in build_router_agent(), any drift
    here would crash the graph with "no edge found" rather than failing
    gracefully. So this function never trusts the raw LLM output directly
    — it normalizes (strip/lower) and validates against VALID_CATEGORIES,
    and anything that doesn't match exactly is forced to
    "unclear_or_other" rather than passed through.
    """
    print(f"\n🔍 Classifying: {state['transcript'][:80]}")

    if not state.get("transcript", "").strip():
        # Empty transcript (e.g. Whisper got silence/noise) — don't even
        # call the LLM, route straight to the fallback.
        return {
            "category": "unclear_or_other",
            "classification_reasoning": "Empty or unintelligible transcript"
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You classify spoken input into exactly one category. "
            "Return ONLY the category name, nothing else — no punctuation, "
            "no explanation, no quotes.\n\n"
            "Categories:\n"
            "- factual_question: stable facts that don't change over time "
            "(capitals, historical dates, scientific constants, definitions)\n"
            "- time_sensitive_question: needs CURRENT information "
            "(prices, scores, news, 'who is the current X', weather)\n"
            "- opinion_question: subjective, no single correct answer "
            "('what's the best X', 'should I do Y')\n"
            "- coding_request: asks to write, explain, review, or debug code\n"
            "- small_talk: greetings, casual conversation, no real question\n"
            "- unclear_or_other: anything that doesn't clearly fit above, "
            "or is too ambiguous/garbled to classify confidently"
        )),
        ("human", "Classify this: {transcript}")
    ])

    llm = get_llm(temperature=0)
    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({"transcript": state["transcript"]})
    except Exception as e:
        print(f"⚠️ Classification LLM call failed: {e}")
        return {
            "category": "unclear_or_other",
            "classification_reasoning": f"Classifier error: {str(e)}"
        }

    # Normalize: strip whitespace, quotes, periods, lowercase
    normalized = raw.strip().strip('"').strip("'").rstrip(".").lower()

    if normalized in VALID_CATEGORIES:
        category = normalized
    else:
        # Hard guard — the exact mismatch case this whole function exists
        # to prevent reaching the graph's conditional edges.
        print(f"⚠️ LLM returned unrecognized category '{raw}' — forcing unclear_or_other")
        category = "unclear_or_other"

    print(f"✅ Category: {category}")
    return {
        "category": category,
        "classification_reasoning": raw
    }

# ── Node 2a — Factual Question ────────────────────────────────────────────────

def handle_factual(state: RouterState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer this factual question accurately and concisely. "
            "If you are not confident about a specific fact (date, number, name), "
            "say so explicitly rather than guessing."
        )),
        ("human", "{transcript}")
    ])
    llm = get_llm(temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"transcript": state["transcript"]})
    return {"response": response, "response_type": "text", "sources": []}

# ── Node 2b — Time-Sensitive Question (Tavily search) ─────────────────────────

def handle_time_sensitive(state: RouterState) -> dict:
    try:
        search = TavilySearchResults(max_results=3)
        results = search.invoke(state["transcript"])
        context = "\n\n".join(
            f"Source: {r['url']}\n{r['content'][:400]}" for r in results
        )
        sources = [r["url"] for r in results]
    except Exception as e:
        print(f"⚠️ Search failed: {e}")
        context = "No search results available."
        sources = []

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Answer using the search results below. Be concise. "
            "If the search results don't actually answer the question, "
            "say so honestly rather than guessing."
        )),
        ("human", "Question: {transcript}\n\nSearch results:\n{context}")
    ])
    llm = get_llm(temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"transcript": state["transcript"], "context": context})
    return {"response": response, "response_type": "text", "sources": sources}

# ── Node 2c — Opinion Question ────────────────────────────────────────────────

def handle_opinion(state: RouterState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "This is a subjective question with no single correct answer. "
            "Present 2-3 different valid perspectives, briefly explain the "
            "reasoning behind each, and avoid asserting one as definitively "
            "'the answer.' Be balanced."
        )),
        ("human", "{transcript}")
    ])
    llm = get_llm(temperature=0.4)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"transcript": state["transcript"]})
    return {"response": response, "response_type": "text", "sources": []}

# ── Node 2d — Coding Request ───────────────────────────────────────────────────

def handle_coding(state: RouterState) -> dict:
    """
    Handles code-related voice requests.
    Response type is "code" — the Streamlit UI renders this with
    st.code() / syntax highlighting instead of plain text, and the
    "Read aloud" button is disabled for this type (see app.py) since
    spoken code is not useful — this is the concrete reason this
    project stayed text-first-with-optional-TTS rather than
    always-speak: code is the clearest case where that would break.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a coding assistant. The user's request came from VOICE "
            "input, so it may contain transcription quirks (e.g. 'def' might "
            "be transcribed oddly, or variable names mis-heard) — interpret "
            "intent charitably. Provide clean, well-commented code, and a "
            "brief plain-English explanation above the code block."
        )),
        ("human", "{transcript}")
    ])
    llm = get_llm(temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"transcript": state["transcript"]})
    return {"response": response, "response_type": "code", "sources": []}

# ── Node 2e — Small Talk ───────────────────────────────────────────────────────

def handle_small_talk(state: RouterState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Respond naturally and briefly to this casual conversational input. Keep it short — 1-2 sentences."),
        ("human", "{transcript}")
    ])
    llm = get_llm(temperature=0.5)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"transcript": state["transcript"]})
    return {"response": response, "response_type": "text", "sources": []}

# ── Node 2f — Unclear / Fallback ──────────────────────────────────────────────

def handle_unclear(state: RouterState) -> dict:
    """
    The catch-all. This node is what makes the system handle "anything"
    safely — every input that the classifier can't confidently place
    elsewhere lands here instead of crashing the graph. Rather than just
    saying "I don't understand," it makes a best-effort attempt to be
    useful, since most "unclear" cases are still genuine attempts at
    communication (mumbled audio, mixed intent, multi-part questions),
    not nonsense.
    """
    transcript = state.get("transcript", "").strip()

    if not transcript:
        return {
            "response": (
                "I didn't catch any speech in that recording — it may have "
                "been silent or too quiet. Could you try recording again?"
            ),
            "response_type": "text",
            "sources": []
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "This input couldn't be confidently classified into a specific "
            "category. Make a genuine best-effort attempt to respond "
            "helpfully — if it seems like a question, try to answer it; if "
            "it's ambiguous, briefly note what you understood and ask a "
            "clarifying question rather than just saying you don't understand."
        )),
        ("human", "{transcript}")
    ])
    llm = get_llm(temperature=0.3)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"transcript": transcript})
    return {"response": response, "response_type": "text", "sources": []}

# ── Routing function ──────────────────────────────────────────────────────────

def route_by_category(state: RouterState) -> str:
    """
    Returns the category string to route on. Defensive default included
    even though classify_input should never leave category unset/invalid
    — belt-and-suspenders against any future code path that might call
    this node directly without going through classify_input first.
    """
    category = state.get("category")
    if category not in VALID_CATEGORIES:
        return "unclear_or_other"
    return category

# ── Build Graph ───────────────────────────────────────────────────────────────

def build_router_agent():
    graph = StateGraph(RouterState)

    graph.add_node("classify_input", classify_input)
    graph.add_node("factual_question", handle_factual)
    graph.add_node("time_sensitive_question", handle_time_sensitive)
    graph.add_node("opinion_question", handle_opinion)
    graph.add_node("coding_request", handle_coding)
    graph.add_node("small_talk", handle_small_talk)
    graph.add_node("unclear_or_other", handle_unclear)

    graph.set_entry_point("classify_input")

    # Every key here MUST exactly match a string in VALID_CATEGORIES —
    # this mapping is the actual enforcement point. If classify_input ever
    # returned something not listed here, LangGraph would raise at runtime
    # rather than silently doing nothing — which is why classify_input's
    # normalization/validation step exists, to make sure that never happens.
    graph.add_conditional_edges(
        "classify_input",
        route_by_category,
        {
            "factual_question": "factual_question",
            "time_sensitive_question": "time_sensitive_question",
            "opinion_question": "opinion_question",
            "coding_request": "coding_request",
            "small_talk": "small_talk",
            "unclear_or_other": "unclear_or_other",
        }
    )

    for node in VALID_CATEGORIES:
        graph.add_edge(node, END)

    return graph.compile()

router_agent = build_router_agent()
