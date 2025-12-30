# Lindorm Search Testing Guide

This directory contains integration tests for Aliyun Lindorm Search vector store.

## Prerequisites

- A running Lindorm Search instance
- Lindorm Search connection details (host, port, username, password)

## Test Files

### 1. `quick_lindorm_test.py` - Quick Connection Test
A standalone script for quick testing of basic Lindorm Search functionality.

**Usage:**

```bash
# Method 1: Edit the file directly
# Open quick_lindorm_test.py and set your credentials in CONFIG section
python tests/vector_stores/quick_lindorm_test.py

# Method 2: Use environment variables
LINDORM_HOST=your-host \
LINDORM_PORT=30070 \
LINDORM_USER=your-username \
LINDORM_PASSWORD=your-password \
python tests/vector_stores/quick_lindorm_test.py
```

**Tests included:**
- Connection to Lindorm Search
- Insert vectors with metadata
- Vector similarity search
- Get vector by ID
- Filtered search by user_id
- Update vector and metadata
- List all vectors
- Hybrid search (vector + text with RRF)
- Collection information
- Delete vector
- Memory class integration

### 2. `test_lindorm_integration.py` - Comprehensive Integration Tests
Full test suite with detailed assertions and error handling.

**Usage:**

```bash
# Set environment variables
export LINDORM_HOST=your-host
export LINDORM_PORT=30070
export LINDORM_USER=your-username
export LINDORM_PASSWORD=your-password

# Run with pytest
pytest tests/vector_stores/test_lindorm_integration.py -v -s

# Or run directly
python tests/vector_stores/test_lindorm_integration.py
```

**Test suites:**
- `TestLindormSearchIntegration` - 11 tests covering all vector store operations
- `TestLindormSearchWithMemory` - Memory class integration tests

### 3. `run_lindorm_test.sh` - Shell Script Runner
Convenient bash script for running tests.

**Usage:**

```bash
# Set environment variables and run
LINDORM_HOST=your-host \
LINDORM_PORT=30070 \
LINDORM_USER=your-username \
LINDORM_PASSWORD=your-password \
./tests/vector_stores/run_lindorm_test.sh
```

### 4. `test_lindorm_search.py` - Unit Tests (Mock)
Unit tests with mocked OpenSearch client, no real connection required.

**Usage:**

```bash
# Run anytime, no credentials needed
pytest tests/vector_stores/test_lindorm_search.py -v
```

## Quick Start Example

```bash
# 1. Set your credentials
export LINDORM_HOST="your-lindorm-host"
export LINDORM_PORT=30070
export LINDORM_USER="your_username"
export LINDORM_PASSWORD="your_password"

# 2. Run quick test
python tests/vector_stores/quick_lindorm_test.py

# 3. Run full integration tests
pytest tests/vector_stores/test_lindorm_integration.py -v -s
```

## What Gets Tested

### Core Operations:
- ✅ Creating collections (indices)
- ✅ Inserting vectors with metadata
- ✅ Vector similarity search (KNN)
- ✅ Hybrid search (vector + text) with RRF fusion
- ✅ Getting vectors by ID
- ✅ Updating vectors and metadata
- ✅ Listing all vectors
- ✅ Deleting vectors
- ✅ Collection management (info, delete)

### Advanced Features:
- ✅ Filtered search (user_id, agent_id, run_id)
- ✅ Hybrid search with RRF (Reciprocal Rank Fusion)
- ✅ Efficient filter queries
- ✅ Routing for user-specific data
- ✅ Custom distance metrics (cosinesimil, l2, innerproduct)

### Integration:
- ✅ Memory class integration
- ✅ Configuration via VectorStoreConfig
- ✅ Factory pattern instantiation
- ✅ OpenSearch client compatibility

## Test Collections

Tests create temporary collections (indices) with names like:
- `test_mem0_20251230_150000`
- `quick_test_20251230_150000`
- `test_mem0_memory_20251230_150000`

These are automatically cleaned up after tests complete via `delete_col()`.

## Lindorm Search Specific Features

### Hybrid Search
Lindorm Search supports hybrid search combining:
- **Vector similarity**: KNN search on embedding vectors
- **Full-text search**: BM25 text search on indexed fields
- **RRF Fusion**: Reciprocal Rank Fusion with configurable weights

Example:
```python
results = vector_store.search(
    query="python programming",  # Text query for BM25
    vectors=[0.1, 0.2, ...],     # Vector query for KNN
    limit=5
)
```

### Distance Metrics
Supported distance methods:
- `cosinesimil` - Cosine similarity (default)
- `l2` - Euclidean distance
- `innerproduct` - Inner product

### Filtered Search
Filter by metadata fields:
```python
results = vector_store.search(
    query="test",
    vectors=[...],
    filters={"user_id": "user1", "agent_id": "agent1"}
)
```

## Troubleshooting

### Connection Refused
- Check LINDORM_HOST is correct
- Verify port is accessible (default: 30070)
- Ensure Lindorm Search instance is running
- Check firewall/security group settings

### Authentication Failed
- Verify username and password
- Check if Lindorm Search requires authentication
- Some instances may not require auth (can omit user/password)

### SSL Errors
- Set `use_ssl=False` and `verify_certs=False` for testing
- Production: Configure proper SSL certificates

### Import Errors
- Install dependencies: `pip install opensearch-py`
- Ensure you're using the correct Python environment

### Collection Already Exists
- Tests auto-generate unique collection names with timestamps
- If manual cleanup needed:
  ```python
  store = LindormSearch(host=..., port=..., collection_name="old_collection")
  store.delete_col()
  ```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| LINDORM_HOST | Yes | - | Lindorm Search instance host |
| LINDORM_PORT | No | 30070 | Lindorm Search port |
| LINDORM_USER | No* | - | Username (if auth enabled) |
| LINDORM_PASSWORD | No* | - | Password (if auth enabled) |

*Required if Lindorm Search has authentication enabled

## Configuration Example

```python
from mem0 import Memory
from mem0.configs.base import MemoryConfig, VectorStoreConfig

config = MemoryConfig(
    vector_store=VectorStoreConfig(
        provider="lindorm_search",
        config={
            "host": "your-lindorm-host",
            "port": 30070,
            "user": "username",  # Optional
            "password": "password",  # Optional
            "collection_name": "mem0",
            "embedding_model_dims": 1536,
            "distance_method": "cosinesimil",
            "use_ssl": False,
            "verify_certs": False
        }
    )
)

memory = Memory(config)
```

## Example Output

```
============================================================
 Lindorm Search - Quick Test
============================================================

============================================================
 1. Testing Connection
============================================================
Host: your-lindorm-host
Port: 30070
Collection: quick_test_20251230_150000
SSL: False
✓ Connected to Lindorm Search successfully!

============================================================
 2. Testing Insert
============================================================
✓ Inserted 3 vectors

============================================================
 3. Testing Search
============================================================
✓ Found 3 results
  1. ID: test_1, Score: 0.9500
  2. ID: test_2, Score: 0.8900
  3. ID: test_3, Score: 0.8500
...
```

## Performance Tips

1. **Batch Operations**: Use bulk insert for better performance
2. **Filters**: Apply filters at search time for efficiency
3. **Hybrid Ratio**: Adjust `hybrid_ratio` in search (0.5 = balanced)
4. **Routing**: Use `user_id` in routing for multi-tenant scenarios

## Contributing

When adding new tests:
1. Add to `test_lindorm_integration.py` for integration tests
2. Add to `test_lindorm_search.py` for unit tests with mocks
3. Ensure cleanup happens in `tearDownClass`
4. Update this README with new test descriptions

## Related Documentation

- [Lindorm Search Documentation](https://www.alibabacloud.com/help/en/lindorm-search/)
- [OpenSearch Python Client](https://opensearch-py.readthedocs.io/)
- [mem0 Documentation](https://docs.mem0.ai/)
