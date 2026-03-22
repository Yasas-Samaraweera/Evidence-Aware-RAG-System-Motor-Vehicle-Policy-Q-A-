"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document

from ..llm.factory import create_chat_model
from ..retrieval.vector_store import retrieve
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool, retrieve_motor_vehicle_tool, retrieve_motorcycle_tool, retrieve_private_car_tool, retrieve_restrictions_tool
from .utils import build_chunk_metadata


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


def _documents_from_tool_message(msg: ToolMessage) -> List[Document]:
    """Extract Document objects from a ToolMessage (content_and_artifact tools).

    LangChain may attach the artifact on ``artifact`` or under
    ``additional_kwargs["artifact"]`` depending on version.
    """
    artifact = getattr(msg, "artifact", None)
    if artifact is None:
        extra = getattr(msg, "additional_kwargs", None)
        if isinstance(extra, dict):
            artifact = extra.get("artifact")
    if not artifact:
        return []
    if isinstance(artifact, list):
        return [doc for doc in artifact if isinstance(doc, Document)]
    if isinstance(artifact, Document):
        return [artifact]
    return []


# Define agents at module level for reuse
# Vehicle-specific retrieval tools use metadata filtering
retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[
        retrieval_tool,
        retrieve_private_car_tool,
        retrieve_motorcycle_tool,
        retrieve_motor_vehicle_tool,
        retrieve_restrictions_tool,
    ],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)


def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from vector store.

    This node:
    - Sends the user's question to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks.
    - Reads the latest ToolMessage (serialized CONTEXT string + artifact Documents).
    - Falls back to a direct vector lookup if artifacts are missing.
    - Builds chunk metadata for citation resolution and stores `state["context"]`.
    """
    question = state["question"]

    result = retrieval_agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])

    context = ""
    retrieved_docs: List[Document] = []

    # Latest ToolMessage wins (most recent tool call); tools use content_and_artifact
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)
            retrieved_docs = _documents_from_tool_message(msg)
            break

    if not retrieved_docs:
        retrieved_docs = retrieve(question, k=4)

    chunk_metadata = build_chunk_metadata(retrieved_docs) if retrieved_docs else {}

    return {
        "context": context,
        "retrieved_documents": retrieved_docs or None,
        "chunk_metadata": chunk_metadata or None,
    }


def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates citation-aware answer from context."""

    question = state["question"]
    context = state.get("context")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    cited_answer = _extract_last_ai_content(messages)

    return {
        # OLD:
        # "draft_answer": draft_answer,

        # NEW (Feature 4):
        "cited_answer": cited_answer,
    }



def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects a citation-aware answer."""

    question = state["question"]
    context = state.get("context", "")
    cited_answer = state.get("cited_answer", "")

    user_content = f"""Question: {question}

Context:
{context}

Cited Answer:
{cited_answer}

Please verify and correct the answer.

Rules:
- Preserve valid citations.
- Remove citations if the statement is removed.
- Add citations if you introduce new information.
- Do NOT leave any factual statement without a citation.
"""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    verified_cited_answer = _extract_last_ai_content(messages)

    return {
        "answer": verified_cited_answer,
    }
