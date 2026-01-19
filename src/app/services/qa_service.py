"""Service layer for handling QA requests.

This module provides a simple interface for the FastAPI layer to interact
with the multi-agent RAG pipeline without depending directly on LangGraph
or agent implementation details.
"""

from typing import Dict, Any, Optional

from ..core.agents.graph import run_qa_flow
from ..core.agents.utils import extract_citations, build_evidence_map


def answer_question(
    question: str,
    vehicle_category: Optional[str] = None,
    restriction_only: bool = False,
) -> Dict[str, Any]:
    """Run the multi-agent QA flow for a given question.

    This function orchestrates the evidence-aware QA pipeline:
    1. Runs the multi-agent QA graph (Retrieval -> Summarization -> Verification)
    2. Extracts citations from the final answer
    3. Builds comprehensive evidence map with chunk content and metadata
    4. Returns answer with rich citation information

    Args:
        question: User's natural language question about the documents.
        vehicle_category: Optional filter for vehicle type:
            - "private_car" or "car" for private cars
            - "motorcycle" or "bike" for motorcycles
            - "motor_vehicle" or "all" for all motor vehicles
            - None for no filter
        restriction_only: If True, only retrieve restriction-related information.

    Returns:
        Dictionary containing:
        - answer: Final verified answer with citations
        - evidence_map: List of evidence objects with chunk content, metadata, etc.
    """
    # Enhance question with vehicle category context if specified
    enhanced_question = question
    if vehicle_category:
        category_context = {
            "private_car": "Focus on private cars and passenger vehicles.",
            "motorcycle": "Focus on motorcycles, motorbikes, and two-wheelers.",
            "motor_vehicle": "Focus on all types of motor vehicles.",
        }
        normalized_category = vehicle_category.lower().strip()
        if normalized_category in ["private_car", "car", "private car"]:
            enhanced_question = f"{category_context['private_car']} {question}"
        elif normalized_category in ["motorcycle", "bike", "motorcycles", "bikes"]:
            enhanced_question = f"{category_context['motorcycle']} {question}"
        elif normalized_category in ["motor_vehicle", "all", "motor vehicles"]:
            enhanced_question = f"{category_context['motor_vehicle']} {question}"
    
    if restriction_only:
        enhanced_question = f"Focus on restrictions and limitations. {enhanced_question}"
    
    # Run multi-agent QA graph
    result = run_qa_flow(enhanced_question)
    answer = result.get("answer", "")
    
    # Get chunk metadata from state (preferred method)
    chunk_metadata = result.get("chunk_metadata")
    
    # Fallback: raw context string for legacy compatibility
    raw_context = result.get("context", "")

    # Extract all cited chunk IDs from the answer (e.g., [doc_p5_c2_abc])
    cited_ids = extract_citations(answer)

    # Build comprehensive evidence map using chunk_metadata (preferred) or raw_context (fallback)
    evidence_map = build_evidence_map(
        cited_ids, 
        chunk_metadata=chunk_metadata,
        raw_context=raw_context if not chunk_metadata else None
    )

    return {
        "answer": answer,
        "evidence_map": evidence_map,
    }
