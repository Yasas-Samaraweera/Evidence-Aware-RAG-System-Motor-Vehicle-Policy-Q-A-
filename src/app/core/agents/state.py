"""LangGraph state schema for the multi-agent QA flow."""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class QAState(TypedDict):
    """State schema for the linear multi-agent QA flow.

    The state flows through three agents:
    1. Retrieval Agent: populates `context` from `question`
    2. Summarization Agent: generates `draft_answer` from `question` + `context`
    3. Verification Agent: produces final `answer` from `question` + `context` + `draft_answer`
    
    Evidence-Aware Extensions:
    - `retrieved_documents`: Full Document objects with metadata for evidence mapping
    - `chunk_metadata`: Mapping of chunk IDs to their metadata and content
    - `evidence_map`: Final evidence map with citations and supporting chunks
    """

    question: str
    context: str | None
    draft_answer: str | None
    answer: str | None

    cited_answer: str | None
    evidence_map: dict[str, dict] | None
    
    # Evidence-aware extensions
    retrieved_documents: Optional[List[Document]]  # Full Document objects from retrieval
    chunk_metadata: Optional[Dict[str, Dict[str, Any]]]  # chunk_id -> {content, page, source, etc.}