from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the documents.
    
    Enhanced with MCP-style vehicle category filtering:
    - vehicle_category: Optional filter for specific vehicle type
    - restriction_only: If True, only retrieve restriction-related information
    """

    question: str
    vehicle_category: Optional[str] = None  # "private_car", "motorcycle", "motor_vehicle", or None
    restriction_only: Optional[bool] = False


class CitationEvidence(BaseModel):
    """Evidence object for a single citation.
    
    This represents a research-grade citation with:
    - Chunk identification (ID, page, source)
    - Content snippet for verification
    - Full metadata for reference
    """
    
    chunk_id: str
    page: Optional[str] = None
    source: Optional[str] = None
    content: Optional[str] = None
    content_snippet: Optional[str] = None
    full_source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QAResponse(BaseModel):
    """Response body for the `/qa` endpoint.

    Evidence-aware response with:
    - Final verified answer with inline citations
    - Comprehensive evidence map with chunk content and metadata
    
    This enables consumers to:
    - Verify answers against source material
    - Display citations with page numbers and sources
    - Show evidence snippets for transparency
    - Link back to original documents
    """

    answer: str
    citations: List[CitationEvidence]
