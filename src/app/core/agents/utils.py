"""Utilities for citation extraction and evidence mapping.

This module provides functions for:
- Extracting citations from answers
- Building chunk metadata from retrieved documents
- Creating comprehensive evidence maps with chunk content and metadata
"""

import re
import os
import hashlib
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


def extract_citations(answer: str) -> List[str]:
    """Extract all chunk IDs from citations in the answer.
    
    Supports multiple citation formats:
    - [chunk_id]
    - [CHUNK_ID: chunk_id]
    - [citation: chunk_id]
    
    Args:
        answer: Answer text containing citations in brackets.
        
    Returns:
        List of unique chunk IDs found in the answer.
    """
    # Pattern 1: Simple brackets [chunk_id]
    simple_citations = re.findall(r"\[([^\]]+)\]", answer)
    
    # Pattern 2: Explicit citation format [CHUNK_ID: chunk_id] or [citation: chunk_id]
    explicit_citations = re.findall(r"\[(?:CHUNK_ID|citation):\s*([^\]]+)\]", answer, re.IGNORECASE)
    
    # Combine and deduplicate
    all_citations = simple_citations + explicit_citations
    
    # Filter out common false positives (like [1], [2] if they're not chunk IDs)
    # Chunk IDs typically contain underscores and alphanumeric characters
    valid_citations = [
        cid.strip() 
        for cid in all_citations 
        if cid.strip() and ('_' in cid or len(cid) > 3)
    ]
    
    return list(set(valid_citations))


def generate_chunk_id(doc: Document, index: int) -> str:
    """Generate a stable, unique chunk ID for a document.
    
    Args:
        doc: Document object with metadata.
        index: Index of the chunk in the retrieval results.
        
    Returns:
        Unique chunk ID in format: source_p{page}_c{index}_{hash}
    """
    page_num = doc.metadata.get("page") or doc.metadata.get("page_number", "unknown")
    source = doc.metadata.get("source", "document")
    source_name = os.path.basename(source)
    
    # Create stable ID
    raw_id = f"{source_name}_{page_num}_{index}"
    chunk_hash = hashlib.md5(raw_id.encode()).hexdigest()[:8]
    chunk_id = f"{source_name}_p{page_num}_c{index}_{chunk_hash}"
    
    return chunk_id


def build_chunk_metadata(documents: List[Document]) -> Dict[str, Dict[str, Any]]:
    """Build a comprehensive metadata mapping for retrieved chunks.
    
    Creates a dictionary mapping chunk IDs to their metadata including:
    - Content (text snippet)
    - Page number
    - Source file
    - Full metadata dictionary
    - Index in retrieval results
    
    Args:
        documents: List of Document objects from retrieval.
        
    Returns:
        Dictionary mapping chunk_id -> {content, page, source, metadata, index}
    """
    chunk_metadata = {}
    
    for idx, doc in enumerate(documents, start=1):
        chunk_id = generate_chunk_id(doc, idx)
        
        page_num = doc.metadata.get("page") or doc.metadata.get("page_number", "unknown")
        source = doc.metadata.get("source", "document")
        source_name = os.path.basename(source)
        
        chunk_metadata[chunk_id] = {
            "chunk_id": chunk_id,
            "content": doc.page_content.strip(),
            "page": page_num,
            "source": source_name,
            "full_source": source,
            "metadata": doc.metadata,
            "index": idx,
        }
    
    return chunk_metadata


def build_evidence_map(
    cited_ids: List[str], 
    chunk_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    raw_context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Build comprehensive evidence map from cited chunk IDs.
    
    This function creates a research-grade evidence map that includes:
    - Chunk ID
    - Page number
    - Source file
    - Content snippet (if available)
    - Full metadata
    
    Args:
        cited_ids: List of chunk IDs cited in the answer.
        chunk_metadata: Optional pre-built chunk metadata mapping (preferred).
        raw_context: Optional raw context string for fallback parsing.
        
    Returns:
        List of evidence dictionaries with citation information.
    """
    evidence = []
    
    # Preferred method: Use chunk_metadata if available
    if chunk_metadata:
        for cid in cited_ids:
            if cid in chunk_metadata:
                evidence.append({
                    "chunk_id": cid,
                    "page": chunk_metadata[cid].get("page"),
                    "source": chunk_metadata[cid].get("source"),
                    "content": chunk_metadata[cid].get("content", ""),
                    "content_snippet": chunk_metadata[cid].get("content", "")[:200] + "..." if len(chunk_metadata[cid].get("content", "")) > 200 else chunk_metadata[cid].get("content", ""),
                    "full_source": chunk_metadata[cid].get("full_source"),
                    "metadata": chunk_metadata[cid].get("metadata", {}),
                })
    
    # Fallback method: Parse from raw_context if chunk_metadata not available
    elif raw_context:
        for cid in cited_ids:
            pattern = rf"\[CHUNK_ID:\s*{re.escape(cid)}\](.*?)(?=\n\n\[CHUNK_ID:|\Z)"
            match = re.search(pattern, raw_context, re.S)
            
            if not match:
                continue
            
            block = match.group(1)
            
            page_match = re.search(r"Page:\s*(.*)", block)
            source_match = re.search(r"Source:\s*(.*)", block)
            
            # Extract content (everything after Source line)
            content_match = re.search(r"Source:\s*.*\n(.*)", block, re.S)
            content = content_match.group(1).strip() if content_match else ""
            
            evidence.append({
                "chunk_id": cid,
                "page": page_match.group(1).strip() if page_match else None,
                "source": source_match.group(1).strip() if source_match else None,
                "content": content,
                "content_snippet": content[:200] + "..." if len(content) > 200 else content,
            })
    
    return evidence
