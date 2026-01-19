"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document

from ..llm.factory import create_chat_model
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


# Define agents at module level for reuse
# Include all vehicle-specific retrieval tools for MCP-style filtering
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
    - Extracts the tool's content (CONTEXT string) from the ToolMessage.
    - Extracts Document objects from tool artifacts for evidence mapping.
    - Builds chunk metadata mapping for citation resolution.
    - Stores the consolidated context string in `state["context"]`.
    
    Evidence-aware enhancement:
    - Preserves full Document objects for evidence mapping
    - Builds comprehensive chunk metadata for citation resolution
    """
    question = state["question"]

    result = retrieval_agent.invoke({"messages": [HumanMessage(content=question)]})

    messages = result.get("messages", [])
    context = ""
    retrieved_docs: List[Document] = []

    # Extract content and artifacts from ToolMessage
    # The retrieval_tool uses response_format="content_and_artifact"
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)
            
            # Try multiple ways to extract artifact (LangChain versions may differ)
            artifact = None
            
            # Method 1: Direct artifact attribute
            if hasattr(msg, "artifact") and msg.artifact:
                artifact = msg.artifact
            # Method 2: Check additional_metadata
            elif hasattr(msg, "additional_kwargs") and isinstance(msg.additional_kwargs, dict):
                artifact = msg.additional_kwargs.get("artifact")
            # Method 3: Check tool_call_id mapping (some versions store artifacts separately)
            elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                # Artifacts might be stored in a separate registry
                pass
            
            # Process artifact if found
            if artifact:
                if isinstance(artifact, list):
                    retrieved_docs = [doc for doc in artifact if isinstance(doc, Document)]
                elif isinstance(artifact, Document):
                    retrieved_docs = [artifact]
            
            break

    # Fallback: If we didn't get documents from artifact, retrieve them directly
    # This ensures we always have Document objects for evidence mapping
    if not retrieved_docs:
        from ..retrieval.vector_store import retrieve
        retrieved_docs = retrieve(question, k=4)

    # Build chunk metadata mapping from retrieved documents
    chunk_metadata = build_chunk_metadata(retrieved_docs) if retrieved_docs else {}

    return {
        "context": context,
        "retrieved_documents": retrieved_docs if retrieved_docs else None,
        "chunk_metadata": chunk_metadata if chunk_metadata else None,
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
