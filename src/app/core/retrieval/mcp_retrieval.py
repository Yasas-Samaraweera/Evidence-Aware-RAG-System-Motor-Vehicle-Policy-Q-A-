"""MCP-style retrieval with metadata filtering for motor vehicle categories.

This module provides Model Context Protocol (MCP) inspired retrieval functions
that filter documents by vehicle type (private cars, motorcycles, all motor vehicles)
and restrictions.
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from ..config import get_settings
from .vector_store import _get_vector_store


def retrieve_with_filter(
    query: str,
    vehicle_category: Optional[str] = None,
    restriction_type: Optional[str] = None,
    k: int = 4,
) -> List[Document]:
    """Retrieve documents with MCP-style metadata filtering.
    
    This function enables filtering by vehicle category and restriction type,
    making it easy to query specific motor vehicle policy information.
    
    Args:
        query: Search query string.
        vehicle_category: Optional filter for vehicle type:
            - "private_car" or "car" - Private cars
            - "motorcycle" or "bike" - Motorcycles
            - "all_vehicles" or "motor_vehicle" - All motor vehicles
            - None - No category filter
        restriction_type: Optional filter for restriction type:
            - "restriction" - Documents about restrictions
            - "policy" - Policy documents
            - None - No restriction filter
        k: Number of documents to retrieve (default: 4).
    
    Returns:
        List of Document objects matching the query and filters.
    """
    vector_store = _get_vector_store()
    
    # Build metadata filter if category or restriction specified
    metadata_filter: Dict[str, Any] = {}
    
    if vehicle_category:
        # Normalize vehicle category
        category_normalized = vehicle_category.lower().strip()
        
        # Map various inputs to standardized categories
        if category_normalized in ["private_car", "car", "private car", "cars"]:
            metadata_filter["vehicle_type"] = "private_car"
        elif category_normalized in ["motorcycle", "bike", "motorcycles", "bikes", "motorbike"]:
            metadata_filter["vehicle_type"] = "motorcycle"
        elif category_normalized in ["all_vehicles", "motor_vehicle", "vehicles", "motor vehicles"]:
            metadata_filter["vehicle_type"] = "motor_vehicle"
    
    if restriction_type:
        metadata_filter["restriction_type"] = restriction_type.lower().strip()
    
    # Create retriever with metadata filter
    if metadata_filter:
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": metadata_filter
            }
        )
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # Enhance query with category context for better semantic matching
    enhanced_query = _enhance_query_with_category(query, vehicle_category)
    
    return retriever.invoke(enhanced_query)


def retrieve_private_cars(
    query: str,
    k: int = 4,
    restriction_only: bool = False,
) -> List[Document]:
    """Retrieve documents specifically about private cars.
    
    Args:
        query: Search query about private cars.
        k: Number of documents to retrieve.
        restriction_only: If True, only return documents about restrictions.
    
    Returns:
        List of Document objects about private cars.
    """
    restriction_type = "restriction" if restriction_only else None
    return retrieve_with_filter(
        query=query,
        vehicle_category="private_car",
        restriction_type=restriction_type,
        k=k,
    )


def retrieve_motorcycles(
    query: str,
    k: int = 4,
    restriction_only: bool = False,
) -> List[Document]:
    """Retrieve documents specifically about motorcycles.
    
    Args:
        query: Search query about motorcycles.
        k: Number of documents to retrieve.
        restriction_only: If True, only return documents about restrictions.
    
    Returns:
        List of Document objects about motorcycles.
    """
    restriction_type = "restriction" if restriction_only else None
    return retrieve_with_filter(
        query=query,
        vehicle_category="motorcycle",
        restriction_type=restriction_type,
        k=k,
    )


def retrieve_motor_vehicles(
    query: str,
    k: int = 4,
    restriction_only: bool = False,
) -> List[Document]:
    """Retrieve documents about all motor vehicles (including restrictions).
    
    Args:
        query: Search query about motor vehicles.
        k: Number of documents to retrieve.
        restriction_only: If True, only return documents about restrictions.
    
    Returns:
        List of Document objects about motor vehicles.
    """
    restriction_type = "restriction" if restriction_only else None
    return retrieve_with_filter(
        query=query,
        vehicle_category="motor_vehicle",
        restriction_type=restriction_type,
        k=k,
    )


def retrieve_restrictions(
    query: str,
    vehicle_category: Optional[str] = None,
    k: int = 4,
) -> List[Document]:
    """Retrieve documents specifically about restrictions.
    
    Args:
        query: Search query about restrictions.
        vehicle_category: Optional filter for specific vehicle type:
            - "private_car" or "car"
            - "motorcycle" or "bike"
            - "motor_vehicle" or "all"
            - None for all vehicle types
        k: Number of documents to retrieve.
    
    Returns:
        List of Document objects about restrictions.
    """
    return retrieve_with_filter(
        query=query,
        vehicle_category=vehicle_category,
        restriction_type="restriction",
        k=k,
    )


def _enhance_query_with_category(query: str, vehicle_category: Optional[str] = None) -> str:
    """Enhance query with vehicle category context for better semantic matching.
    
    Since metadata filtering might not always be reliable if metadata wasn't
    properly indexed, we enhance the query with category keywords to improve
    semantic search results.
    
    Args:
        query: Original query string.
        vehicle_category: Optional vehicle category.
    
    Returns:
        Enhanced query string with category context.
    """
    if not vehicle_category:
        return query
    
    category_normalized = vehicle_category.lower().strip()
    
    # Add category-specific context to query
    if category_normalized in ["private_car", "car", "private car", "cars"]:
        context = "private car private vehicle passenger car"
    elif category_normalized in ["motorcycle", "bike", "motorcycles", "bikes", "motorbike"]:
        context = "motorcycle motorbike bike two-wheeler"
    elif category_normalized in ["all_vehicles", "motor_vehicle", "vehicles", "motor vehicles"]:
        context = "motor vehicle automotive vehicle car motorcycle"
    else:
        return query
    
    return f"{query} {context}"
