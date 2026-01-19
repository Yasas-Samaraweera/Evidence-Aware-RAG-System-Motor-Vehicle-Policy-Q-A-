"""Vector store wrapper for Pinecone integration with LangChain."""

from pathlib import Path
from functools import lru_cache
from typing import List

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from ..config import get_settings


@lru_cache(maxsize=1)
def _get_vector_store() -> PineconeVectorStore:
    """Create a PineconeVectorStore instance configured from settings."""
    settings = get_settings()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model_name,
        api_key=settings.openai_api_key,
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

def get_retriever(k: int | None = None):
    """Get a Pinecone retriever instance.

    Args:
        k: Number of documents to retrieve (defaults to config value).

    Returns:
        PineconeVectorStore instance configured as a retriever.
    """
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k

    vector_store = _get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int | None = None) -> List[Document]:
    """Retrieve documents from Pinecone for a given query.

    Args:
        query: Search query string.
        k: Number of documents to retrieve (defaults to config value).

    Returns:
        List of Document objects with metadata (including page numbers).
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(query)

def _extract_vehicle_metadata(content: str) -> dict:
    """Extract vehicle category metadata from document content.
    
    This function analyzes the text content to determine which vehicle
    categories the document relates to, enabling MCP-style filtering.
    
    Args:
        content: Text content of the document chunk.
        
    Returns:
        Dictionary with metadata about vehicle types and restrictions.
    """
    content_lower = content.lower()
    metadata = {}
    
    # Check for vehicle types
    private_car_keywords = [
        "private car", "private vehicle", "passenger car", "car", "cars",
        "sedan", "hatchback", "suv", "automobile", "motor car"
    ]
    motorcycle_keywords = [
        "motorcycle", "motorbike", "bike", "bikes", "two-wheeler",
        "two wheeler", "motorcycle", "scooter", "motor cycle"
    ]
    motor_vehicle_keywords = [
        "motor vehicle", "vehicle", "vehicles", "automotive",
        "all vehicles", "motor vehicles"
    ]
    restriction_keywords = [
        "restriction", "restrict", "prohibited", "prohibition", "not allowed",
        "forbidden", "ban", "limited", "limit", "regulation", "regulate",
        "constraint", "not permitted", "do not"
    ]
    
    # Determine vehicle type
    has_private_car = any(keyword in content_lower for keyword in private_car_keywords)
    has_motorcycle = any(keyword in content_lower for keyword in motorcycle_keywords)
    has_motor_vehicle = any(keyword in content_lower for keyword in motor_vehicle_keywords)
    
    # Set vehicle_type metadata (can be multiple, but prioritize specificity)
    if has_private_car and not has_motorcycle:
        metadata["vehicle_type"] = "private_car"
    elif has_motorcycle and not has_private_car:
        metadata["vehicle_type"] = "motorcycle"
    elif has_motor_vehicle or (has_private_car and has_motorcycle):
        metadata["vehicle_type"] = "motor_vehicle"
    
    # Check for restrictions
    has_restriction = any(keyword in content_lower for keyword in restriction_keywords)
    if has_restriction:
        metadata["restriction_type"] = "restriction"
    
    return metadata


def index_documents(file_path: Path) -> int:
    """Index a PDF file into the Pinecone vector store.

    This function loads a PDF, splits it into chunks, extracts vehicle
    category metadata, and indexes everything into Pinecone for MCP-style
    filtering.

    Args:
        file_path: Path to the PDF file to index.

    Returns:
        The number of document chunks indexed.
    """
    loader = PyPDFLoader(str(file_path), mode="single")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    
    # Add vehicle metadata to each chunk
    for doc in texts:
        vehicle_metadata = _extract_vehicle_metadata(doc.page_content)
        # Merge extracted metadata with existing metadata
        doc.metadata.update(vehicle_metadata)

    vector_store = _get_vector_store()
    vector_store.add_documents(texts)
    return len(texts)