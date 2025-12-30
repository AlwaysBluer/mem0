# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Lint Commands

### Environment Setup
```bash
make install              # Create hatch environment
make install_all          # Install all optional dependencies
```

### Code Quality
```bash
make format               # Format code with ruff
make sort                 # Sort imports with isort
make lint                 # Run ruff linter
make lint-fix             # Auto-fix linting issues
```

### Testing
```bash
make test                 # Run all tests with pytest
make test-py-3.9          # Run tests on Python 3.9
make test-py-3.10         # Run tests on Python 3.10
make test-py-3.11         # Run tests on Python 3.11
make test-py-3.12         # Run tests on Python 3.12
```

### Direct hatch commands
```bash
hatch run format          # Format code
hatch run lint            # Lint code
hatch run test tests/     # Run tests
```

### Build and Publish
```bash
make build                # Build package
make publish              # Publish to PyPI
```

## Architecture Overview

### Core Components

**Memory Layer** (`mem0/memory/`)
- `main.py` - Core `Memory` and `AsyncMemory` classes implementing CRUD operations
- `base.py` - Abstract base class defining the memory interface
- `storage.py` - `SQLiteManager` for persistent history storage
- `graph_memory.py` - Graph-based memory implementation
- `kuzu_memory.py`, `memgraph_memory.py` - Graph database-specific implementations

**Factory Pattern** (`mem0/utils/factory.py`)
- Centralized instantiation of pluggable components:
  - `LlmFactory` - LLM providers (OpenAI, Anthropic, Azure, Groq, etc.)
  - `EmbedderFactory` - Embedding models
  - `VectorStoreFactory` - Vector databases (Qdrant, Chroma, Pinecone, etc.)
  - `GraphStoreFactory` - Graph databases (Neo4j, Memgraph, Kuzu, Neptune)
  - `RerankerFactory` - Reranking implementations

**Configuration** (`mem0/configs/`)
- `base.py` - `MemoryConfig` (root), `MemoryItem` (data model)
- `embeddings/`, `llms/`, `vector_stores/`, `rerankers/` - Provider-specific configs
- `prompts.py` - LLM prompts for fact extraction and memory updates

**Pluggable Subsystems**
- `mem0/llms/` - LLM provider implementations
- `mem0/embeddings/` - Embedding model implementations
- `mem0/vector_stores/` - Vector database implementations
- `mem0/reranker/` - Reranking implementations
- `mem0/graphs/` - Graph store integrations

### Client Layer (`mem0/client/`)
- `main.py` - `MemoryClient` and `AsyncMemoryClient` for platform API interaction
- `project.py` - Project management operations

### Key Design Patterns

1. **Factory Pattern**: All major components use factories for instantiation
2. **Pydantic Models**: Configuration uses Pydantic `BaseModel` for validation
3. **Async Support**: Both sync and async versions of Memory and Client classes
4. **Provider Abstraction**: Unified interfaces over multiple backend implementations

### Memory Types
- **User Memory**: User-specific preferences and information
- **Session Memory**: Conversation-level context (ephemeral)
- **Agent Memory**: Agent-specific knowledge and behaviors

### Data Flow
1. `Memory.add()` receives messages/conversation
2. LLM extracts facts using configurable prompt
3. Embedder generates embeddings
4. Vector store stores + retrieves by similarity
5. Reranker optionally reorders results
6. Optional graph store extracts relationships

## Important Notes

- Default LLM is `gpt-4.1-nano-2025-04-14` (OpenAI)
- Default vector store is Qdrant
- History DB defaults to `~/.mem0/history.db`
- Telemetry via PostHog (can be disabled)
- Ruff line length: 120 characters
- Python version support: 3.9-3.12
- Comments MUST be in English
- Never commit `.env` files or API keys
