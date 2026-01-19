"""Utilities for serializing retrieved document chunks."""

from typing import List
from langchain_core.documents import Document
import os
import hashlib


def serialize_chunks(docs: List[Document]) -> str:
    """Serialize Document objects into citation-aware CONTEXT."""

    context_parts = []

    for idx, doc in enumerate(docs, start=1):
        # --- Metadata ---
        page_num = doc.metadata.get("page") or doc.metadata.get("page_number", "unknown")
        source = doc.metadata.get("source", "document")

        # Make source name clean (file only)
        source_name = os.path.basename(source)

        # Stable + unique chunk id
        raw_id = f"{source_name}_{page_num}_{idx}"
        chunk_hash = hashlib.md5(raw_id.encode()).hexdigest()[:8]
        chunk_id = f"{source_name}_p{page_num}_c{idx}_{chunk_hash}"

        # --- Format ---
        block = (
            f"[CHUNK_ID: {chunk_id}]\n"
            f"Page: {page_num}\n"
            f"Source: {source_name}\n"
            f"{doc.page_content.strip()}"
        )

        context_parts.append(block)

    return "\n\n".join(context_parts)
