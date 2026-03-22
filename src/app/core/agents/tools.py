"""Tools available to agents in the multi-agent RAG system."""

from langchain_core.tools import tool

from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks
from ..retrieval.filtered_retrieval import (
    retrieve_private_cars,
    retrieve_motorcycles,
    retrieve_motor_vehicles,
    retrieve_restrictions,
)


@tool(response_format="content_and_artifact")
def retrieval_tool(query: str):
    """Search the vector database for relevant document chunks.

    This tool retrieves the top 4 most relevant chunks from the Pinecone
    vector store based on the query. The chunks are formatted with page
    numbers and indices for easy reference.

    Args:
        query: The search query string to find relevant document chunks.

    Returns:
        Tuple of (serialized_content, artifact) where:
        - serialized_content: A formatted string containing the retrieved chunks
          with metadata. Format: "Chunk 1 (page=X): ...\n\nChunk 2 (page=Y): ..."
        - artifact: List of Document objects with full metadata for reference
    """
    # Retrieve documents from vector store
    docs = retrieve(query, k=4)

    # Serialize chunks into formatted string (content)
    context = serialize_chunks(docs)

    # Return tuple: (serialized content, artifact documents)
    # This follows LangChain's content_and_artifact response format
    return context, docs


@tool(response_format="content_and_artifact")
def retrieve_private_car_tool(query: str, restriction_only: bool = False):
    """Retrieve documents specifically about private cars and their restrictions.

    Use this tool when the question is about private cars, passenger vehicles,
    or private car policies and restrictions.

    Args:
        query: The search query about private cars.
        restriction_only: If True, only retrieve documents about restrictions.
                         Default is False.

    Returns:
        Tuple of (serialized_content, artifact) with private car information.
    """
    docs = retrieve_private_cars(query, k=4, restriction_only=restriction_only)
    context = serialize_chunks(docs)
    return context, docs


@tool(response_format="content_and_artifact")
def retrieve_motorcycle_tool(query: str, restriction_only: bool = False):
    """Retrieve documents specifically about motorcycles and their restrictions.

    Use this tool when the question is about motorcycles, motorbikes, bikes,
    two-wheelers, or motorcycle policies and restrictions.

    Args:
        query: The search query about motorcycles.
        restriction_only: If True, only retrieve documents about restrictions.
                         Default is False.

    Returns:
        Tuple of (serialized_content, artifact) with motorcycle information.
    """
    docs = retrieve_motorcycles(query, k=4, restriction_only=restriction_only)
    context = serialize_chunks(docs)
    return context, docs


@tool(response_format="content_and_artifact")
def retrieve_motor_vehicle_tool(query: str, restriction_only: bool = False):
    """Retrieve documents about all motor vehicles and their restrictions.

    Use this tool when the question is about motor vehicles in general,
    automotive vehicles, or comprehensive vehicle policies and restrictions.

    Args:
        query: The search query about motor vehicles.
        restriction_only: If True, only retrieve documents about restrictions.
                         Default is False.

    Returns:
        Tuple of (serialized_content, artifact) with motor vehicle information.
    """
    docs = retrieve_motor_vehicles(query, k=4, restriction_only=restriction_only)
    context = serialize_chunks(docs)
    return context, docs


@tool(response_format="content_and_artifact")
def retrieve_restrictions_tool(query: str, vehicle_type: str = "all"):
    """Retrieve documents specifically about vehicle restrictions.

    Use this tool when the question focuses on restrictions, limitations,
    prohibitions, or regulatory constraints for vehicles.

    Args:
        query: The search query about restrictions.
        vehicle_type: Type of vehicle to filter by:
                     - "car" or "private_car" for private cars
                     - "motorcycle" or "bike" for motorcycles
                     - "all" or "motor_vehicle" for all vehicles
                     Default is "all".

    Returns:
        Tuple of (serialized_content, artifact) with restriction information.
    """
    vehicle_category = None if vehicle_type.lower() == "all" else vehicle_type
    docs = retrieve_restrictions(query, vehicle_category=vehicle_category, k=4)
    context = serialize_chunks(docs)
    return context, docs
