# Evidence-Aware RAG System with Motor Vehicle Policy Q&A

A research-grade Retrieval-Augmented Generation (RAG) system that provides evidence-aware answers with chunk citations, featuring a multi-agent pipeline, MCP-style metadata filtering, and an interactive web UI.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Workflow & Process](#workflow--process)
- [Components](#components)
- [API Endpoints](#api-endpoints)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Technical Details](#technical-details)

## 🎯 Overview

This system combines:
- **Multi-Agent RAG Pipeline**: Three specialized agents (Retrieval, Summarization, Verification) working together
- **Evidence-Aware Answers**: Every answer includes inline citations that link back to source chunks
- **MCP-Style Filtering**: Model Context Protocol-inspired metadata filtering for vehicle categories
- **Interactive UI**: Modern web interface with citation exploration, evidence panel, and chunk viewer
- **Vector Database**: Pinecone integration for semantic search

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Answer View  │  │Evidence Panel│  │Chunk Viewer  │      │
│  │ [1] [2] [3]  │  │ Citation Table│  │ Full Content │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP POST /qa
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          QA Service Layer                            │   │
│  │  - Question Validation                               │   │
│  │  - Citation Extraction                               │   │
│  │  - Evidence Map Building                             │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Agent RAG Pipeline (LangGraph)             │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Retrieval   │───▶│Summarization │───▶│ Verification │  │
│  │    Agent     │    │    Agent     │    │    Agent     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│        │                    │                    │           │
│        │ Uses Tools         │ Generates          │ Verifies  │
│        ▼                    ▼                    ▼           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Specialized Retrieval Tools                │   │
│  │  - retrieve_private_car_tool                         │   │
│  │  - retrieve_motorcycle_tool                          │   │
│  │  - retrieve_motor_vehicle_tool                       │   │
│  │  - retrieve_restrictions_tool                        │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP Retrieval Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Metadata Filtering by Vehicle Category              │   │
│  │  - private_car                                       │   │
│  │  - motorcycle                                        │   │
│  │  - motor_vehicle                                     │   │
│  │  - restriction_type                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Pinecone Vector Store (Embeddings)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - OpenAI Embeddings (text-embedding-3-large)       │   │
│  │  - Document Chunks with Metadata                     │   │
│  │  - Vehicle Category Tags                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 1. Evidence-Aware Answers
- **Inline Citations**: Every factual statement includes citations like [1], [2]
- **Verifiable Claims**: Users can verify every claim against source material
- **Chunk Content**: Full source text available for each citation
- **Page Numbers**: Direct reference to document pages
- **Source Files**: Track which document each claim comes from

### 2. Multi-Agent Pipeline
- **Retrieval Agent**: Intelligently searches vector database using specialized tools
- **Summarization Agent**: Generates citation-aware answers from retrieved context
- **Verification Agent**: Validates and corrects citations, removes hallucinations

### 3. MCP-Style Filtering
- **Vehicle Categories**: Filter by private cars, motorcycles, or all vehicles
- **Restriction Filtering**: Focus on restriction-related information
- **Metadata-Based**: Uses document metadata for precise filtering
- **Semantic Enhancement**: Combines metadata filtering with semantic search

### 4. Interactive UI
- **Answer View**: Clickable citation badges
- **Evidence Panel**: Table showing all citations with page and source
- **Chunk Viewer**: Displays full chunk content for selected citations
- **Debug Mode**: Toggle to show technical details
- **Real-time Updates**: Citations and evidence sync across components

## 🔄 Workflow & Process

### End-to-End Flow

#### Step 1: User Question
```
User enters: "What are the restrictions for private cars?"
Vehicle Category: "private_car"
Restrictions Only: true
```

#### Step 2: API Request Processing
```
POST /qa
{
  "question": "What are the restrictions for private cars?",
  "vehicle_category": "private_car",
  "restriction_only": true
}
```

#### Step 3: QA Service Layer
- Validates question
- Enhances query with vehicle category context
- Initializes multi-agent pipeline

#### Step 4: Retrieval Agent
```
Retrieval Agent receives enhanced question:
"Focus on private cars and passenger vehicles. 
 What are the restrictions for private cars?"
```

**Agent Actions:**
1. Selects appropriate tool: `retrieve_private_car_tool`
2. Tool calls MCP retrieval with filters:
   - `vehicle_category: "private_car"`
   - `restriction_type: "restriction"`
3. Retrieves top 4 relevant chunks from Pinecone
4. Chunks include metadata:
   - `vehicle_type: "private_car"`
   - `restriction_type: "restriction"`
   - `page: "14"`
   - `source: "motor-english-policy-book-2023.pdf"`
5. Serializes chunks with chunk IDs:
   ```
   [CHUNK_ID: motor-english-policy-book-2023_p14_c1_a1b2c3d4]
   Page: 14
   Source: motor-english-policy-book-2023.pdf
   [Chunk content...]
   ```

#### Step 5: Summarization Agent
```
Summarization Agent receives:
- Original question
- Retrieved context with chunk IDs
```

**Agent Actions:**
1. Generates answer using ONLY the provided context
2. Includes citations after each factual statement:
   ```
   Private cars have several restrictions [motor-english-policy-book-2023_p14_c1_a1b2c3d4]. 
   These include speed limits [motor-english-policy-book-2023_p14_c2_b2c3d4e5] 
   and parking regulations [motor-english-policy-book-2023_p15_c1_c3d4e5f6].
   ```
3. Ensures every claim has at least one citation
4. Does not invent citations

#### Step 6: Verification Agent
```
Verification Agent receives:
- Original question
- Full context
- Draft answer with citations
```

**Agent Actions:**
1. Cross-references every claim against context
2. Validates that chunk IDs exist in context
3. Removes invalid citations
4. Adds missing citations for unsupported claims
5. Ensures no uncited factual statements remain

#### Step 7: Citation Extraction & Evidence Mapping
```
Backend extracts citations from answer:
["motor-english-policy-book-2023_p14_c1_a1b2c3d4",
 "motor-english-policy-book-2023_p14_c2_b2c3d4e5",
 "motor-english-policy-book-2023_p15_c1_c3d4e5f6"]
```

**Evidence Map Building:**
1. Looks up each chunk ID in chunk metadata
2. Retrieves:
   - Full chunk content
   - Page number
   - Source file
   - Full metadata
3. Creates evidence objects:
   ```json
   {
     "chunk_id": "motor-english-policy-book-2023_p14_c1_a1b2c3d4",
     "page": "14",
     "source": "motor-english-policy-book-2023.pdf",
     "content": "Full chunk text...",
     "content_snippet": "First 200 chars...",
     "full_source": "data/uploads/motor-english-policy-book-2023.pdf",
     "metadata": {...}
   }
   ```

#### Step 8: Response to UI
```json
{
  "answer": "Private cars have several restrictions [1]. These include...",
  "citations": [
    {
      "chunk_id": "motor-english-policy-book-2023_p14_c1_a1b2c3d4",
      "page": "14",
      "source": "motor-english-policy-book-2023.pdf",
      "content": "Full chunk content...",
      ...
    },
    ...
  ]
}
```

#### Step 9: UI Rendering
1. **Answer View**: 
   - Replaces chunk IDs with numbered citations [1], [2], [3]
   - Makes citations clickable
   
2. **Evidence Panel**:
   - Populates table with all citations
   - Shows chunk ID, page, source
   
3. **Chunk Viewer**:
   - Ready to display chunk content when citation clicked

#### Step 10: User Interaction
- User clicks citation [1] in answer
- Chunk viewer shows full content
- Evidence panel highlights corresponding row
- User can verify claim against source

## 📦 Components

### 1. Frontend (Static HTML/CSS/JavaScript)

**File**: `static/index.html`

**Components**:
- Search form with vehicle category filters
- Answer display with inline citations
- Evidence panel table
- Chunk viewer
- Debug mode toggle

**Key Functions**:
- `fetchAnswer()`: Calls API and processes response
- `displayAnswer()`: Renders answer with clickable citations
- `displayEvidencePanel()`: Populates evidence table
- `showChunk()`: Displays chunk content in viewer

### 2. API Layer (FastAPI)

**File**: `src/app/api.py`

**Endpoints**:
- `POST /qa`: Question answering with evidence
- `POST /index-pdf`: PDF indexing endpoint
- `GET /`: Serves UI

**Key Features**:
- Request validation
- Response model conversion
- Error handling
- CORS support

### 3. Service Layer

**File**: `src/app/services/qa_service.py`

**Function**: `answer_question()`

**Responsibilities**:
- Orchestrates multi-agent pipeline
- Extracts citations from answer
- Builds evidence map
- Returns structured response

### 4. Multi-Agent Pipeline

**Files**:
- `src/app/core/agents/graph.py`: LangGraph orchestration
- `src/app/core/agents/agents.py`: Agent node implementations
- `src/app/core/agents/prompts.py`: System prompts
- `src/app/core/agents/tools.py`: Agent tools
- `src/app/core/agents/state.py`: State schema

**Flow**:
```
START → retrieval_node → summarization_node → verification_node → END
```

### 5. Retrieval Layer

**Files**:
- `src/app/core/retrieval/vector_store.py`: Pinecone integration
- `src/app/core/retrieval/filtered_retrieval.py`: metadata filtering for vehicle categories
- `src/app/core/retrieval/serialization.py`: Chunk serialization

**Features**:
- Semantic search with OpenAI embeddings
- Metadata filtering by vehicle category
- Chunk ID generation
- Content serialization

### 6. Citation & Evidence Utilities

**File**: `src/app/core/agents/utils.py`

**Functions**:
- `extract_citations()`: Finds citations in answer text
- `build_chunk_metadata()`: Creates metadata mapping
- `build_evidence_map()`: Resolves citations to evidence
- `generate_chunk_id()`: Creates stable chunk IDs

### 7. Models

**File**: `src/app/models.py`

**Models**:
- `QuestionRequest`: API request schema
- `CitationEvidence`: Citation object schema
- `QAResponse`: API response schema

## 🔌 API Endpoints

### POST /qa

**Description**: Submit a question and get an evidence-aware answer

**Request Body**:
```json
{
  "question": "What are the restrictions for private cars?",
  "vehicle_category": "private_car",  // Optional: "private_car", "motorcycle", "motor_vehicle", or null
  "restriction_only": true  // Optional: boolean, default false
}
```

**Response**:
```json
{
  "answer": "Private cars have several restrictions [1] [2]...",
  "citations": [
    {
      "chunk_id": "motor-english-policy-book-2023_p14_c1_a1b2c3d4",
      "page": "14",
      "source": "motor-english-policy-book-2023.pdf",
      "content": "Full chunk content...",
      "content_snippet": "First 200 characters...",
      "full_source": "data/uploads/motor-english-policy-book-2023.pdf",
      "metadata": {...}
    }
  ]
}
```

### POST /index-pdf

**Description**: Upload and index a PDF file

**Request**: Multipart form data with PDF file

**Response**:
```json
{
  "filename": "motor-english-policy-book-2023.pdf",
  "chunks_indexed": 245,
  "message": "PDF indexed successfully."
}
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- Pinecone account and API key
- OpenAI API key

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd class-12
```

### Step 2: Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Create `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-large

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here

# Retrieval Configuration
RETRIEVAL_K=4
```

### Step 4: Create Pinecone Index

```python
from pinecone import Pinecone
pc = Pinecone(api_key="your_api_key")
pc.create_index(
    name="your_index_name",
    dimension=3072,  # For text-embedding-3-large
    metric="cosine"
)
```

### Step 5: Index Documents

**Via API**:
```bash
curl -X POST http://localhost:8000/index-pdf \
  -F "file=@data/uploads/motor-english-policy-book-2023.pdf"
```

**Or via Python**:
```python
from src.app.services.indexing_service import index_pdf_file
from pathlib import Path

chunks = index_pdf_file(Path("data/uploads/motor-english-policy-book-2023.pdf"))
print(f"Indexed {chunks} chunks")
```

### Step 6: Run Server

```bash
uvicorn src.app.api:app --reload --port 8000
```

### Step 7: Access UI

Open browser: `http://localhost:8000`

## 📖 Usage Guide

### Basic Usage

1. **Open UI**: Navigate to `http://localhost:8000`

2. **Enter Question**:
   ```
   What are the restrictions for private cars?
   ```

3. **Select Filters** (Optional):
   - Vehicle Category: "Private Cars"
   - Check "Restrictions Only"

4. **Click "Get Answer"**

5. **View Results**:
   - Answer with inline citations [1] [2]
   - Evidence panel with all citations
   - Click citations to view chunk content

### Advanced Usage

#### Filtering by Vehicle Type

**Private Cars**:
```json
{
  "question": "What are the speed limits?",
  "vehicle_category": "private_car"
}
```

**Motorcycles**:
```json
{
  "question": "What are the helmet requirements?",
  "vehicle_category": "motorcycle"
}
```

**All Vehicles**:
```json
{
  "question": "What are the general regulations?",
  "vehicle_category": "motor_vehicle"
}
```

#### Restriction-Only Queries

```json
{
  "question": "What are prohibited?",
  "restriction_only": true
}
```

### UI Features

#### Citation Interaction
- **Click citation badge** [1] → View chunk content
- **Click evidence row** → View chunk content
- **Hover citation** → See citation number
- **Active citation** → Highlighted in red

#### Debug Mode
- **Toggle debug mode** → Shows:
  - Chunk IDs
  - Index numbers
  - Full source paths
  - Metadata JSON
  - Query term highlighting

#### Evidence Panel
- **Sortable columns**: Citation, Page, Source
- **Hover rows**: Highlight effect
- **Click rows**: Show chunk content
- **Selected row**: Blue highlight

#### Chunk Viewer
- **Full content**: Complete chunk text
- **Query highlighting**: Matched terms highlighted
- **Metadata**: Page number, source file
- **Scroll support**: For long chunks

## 🔧 Technical Details

### Citation Format

**In Answer Text**:
```
[chunk_id] or [CHUNK_ID: chunk_id]
```

**Example**:
```
Private cars are restricted [motor-english-policy-book-2023_p14_c1_a1b2c3d4].
```

**In UI**:
```
Private cars are restricted [1].
```

### Chunk ID Format

```
{source_name}_p{page}_c{index}_{hash}
```

**Example**:
```
motor-english-policy-book-2023_p14_c1_a1b2c3d4
```

**Components**:
- `motor-english-policy-book-2023`: Source filename
- `p14`: Page number
- `c1`: Chunk index
- `a1b2c3d4`: MD5 hash (first 8 chars)

### Metadata Structure

**Document Chunks Include**:
```python
{
    "page": "14",
    "source": "motor-english-policy-book-2023.pdf",
    "vehicle_type": "private_car",  # Auto-detected
    "restriction_type": "restriction",  # Auto-detected
    ...
}
```

**Evidence Objects Include**:
```python
{
    "chunk_id": "...",
    "page": "14",
    "source": "motor-english-policy-book-2023.pdf",
    "content": "Full chunk text...",
    "content_snippet": "First 200 chars...",
    "full_source": "data/uploads/...",
    "metadata": {...},
    "index": 1  # Citation number
}
```

### Agent Prompts

#### Retrieval Agent
- Uses specialized tools based on vehicle category
- Consolidates context from multiple tool calls
- Formats context with chunk IDs

#### Summarization Agent
- Generates citation-aware answers
- Includes citations after each claim
- Only uses provided context

#### Verification Agent
- Validates citations against context
- Removes invalid citations
- Adds missing citations
- Ensures no uncited claims

### Retrieval Tools

1. **retrieve_private_car_tool**: Private car documents
2. **retrieve_motorcycle_tool**: Motorcycle documents
3. **retrieve_motor_vehicle_tool**: All motor vehicles
4. **retrieve_restrictions_tool**: Restriction documents
5. **retrieval_tool**: General retrieval (fallback)

### MCP Filtering Logic

**Filter Types**:
- `vehicle_type`: "private_car", "motorcycle", "motor_vehicle"
- `restriction_type`: "restriction"

**Filter Application**:
1. Check document metadata for vehicle_type
2. Filter chunks by restriction_type if requested
3. Enhance query with category keywords
4. Perform semantic search on filtered set

### State Management

**QAState Schema**:
```python
{
    "question": str,
    "context": str | None,
    "answer": str | None,
    "cited_answer": str | None,
    "retrieved_documents": List[Document] | None,
    "chunk_metadata": Dict[str, Dict] | None,
    "evidence_map": Dict | None
}
```

**State Flow**:
```
Initial State: {question: "..."}
    ↓
Retrieval Node: Adds {context, retrieved_documents, chunk_metadata}
    ↓
Summarization Node: Adds {cited_answer}
    ↓
Verification Node: Adds {answer}
    ↓
Final State: Complete state with all fields
```

## 🎓 Research-Grade Features

### 1. Evidence Traceability
Every claim can be traced back to:
- Specific document chunk
- Page number
- Source file
- Full content for verification

### 2. Citation Validation
- Automated verification of citations
- Removes invalid citations
- Ensures comprehensive coverage

### 3. Hallucination Prevention
- Three-agent verification pipeline
- Context grounding
- Citation requirements

### 4. Transparency
- Debug mode for technical details
- Full metadata access
- Query term highlighting

## 🔍 Debugging

### Console Logging

The UI includes console logging for debugging:

```javascript
console.log('API Response:', data);
console.log('Citations:', data.citations);
console.log('Citation map:', currentCitations);
```

### Debug Mode

Toggle debug mode in UI to see:
- Chunk IDs
- Metadata
- Index numbers
- Full source paths

### Common Issues

**No citations found**:
- Check if answer includes citation format
- Verify chunk IDs match between answer and evidence map
- Check console logs for mapping issues

**Chunk viewer empty**:
- Verify chunk_id matches citation
- Check if content exists in evidence map
- Use debug mode to see available citations

**Evidence panel empty**:
- Verify citations array in API response
- Check citation extraction logic
- Ensure chunk_metadata is populated

## 📚 References

- **LangChain**: https://python.langchain.com/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **Pinecone**: https://www.pinecone.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. Maintain citation integrity
2. Ensure evidence traceability
3. Add tests for new features
4. Update documentation

## 📄 License

[Your License Here]

## 👥 Authors

[Your Name/Team]

---

**Built with ❤️ for Research-Grade RAG Systems**
