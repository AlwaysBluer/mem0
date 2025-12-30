"""
Quick Lindorm Search Connection Test

This script tests basic Lindorm Search connectivity and operations.
Set your connection details in the CONFIG section below.
"""

import os
from datetime import datetime
from mem0.vector_stores.lindorm_search import LindormSearch

# ===================== CONFIGURATION =====================
# Set your Lindorm Search connection details here
CONFIG = {
    "host": os.getenv("LINDORM_HOST", "YOUR_LINDORM_HOST"),  # Change this
    "port": int(os.getenv("LINDORM_PORT", 30070)),
    "user": os.getenv("LINDORM_USER", ""),  # Change if needed
    "password": os.getenv("LINDORM_PASSWORD", ""),  # Change if needed
    "collection_name": f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "embedding_model_dims": 1536,
    "distance_method": "cosinesimil",
    "use_ssl": False,
    "verify_certs": False
}
# ========================================================

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def test_connection():
    """Test basic connection to Lindorm Search."""
    print_section("1. Testing Connection")
    print(f"Host: {CONFIG['host']}")
    print(f"Port: {CONFIG['port']}")
    print(f"Collection: {CONFIG['collection_name']}")
    print(f"SSL: {CONFIG['use_ssl']}")
    
    try:
        # Remove user/password if empty
        config = CONFIG.copy()
        if not config["user"]:
            config.pop("user", None)
        if not config["password"]:
            config.pop("password", None)
        
        vector_store = LindormSearch(**config)
        print("✓ Connected to Lindorm Search successfully!")
        return vector_store
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_insert(vector_store):
    """Test inserting vectors."""
    print_section("2. Testing Insert")
    
    vectors = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
    payloads = [
        {"data": "First test memory", "user_id": "user1"},
        {"data": "Second test memory", "user_id": "user2"},
        {"data": "Third test memory", "user_id": "user1"}
    ]
    ids = ["test_1", "test_2", "test_3"]
    
    try:
        results = vector_store.insert(vectors=vectors, payloads=payloads, ids=ids)
        print(f"✓ Inserted {len(results)} vectors")
        return ids
    except Exception as e:
        print(f"✗ Insert failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_search(vector_store):
    """Test searching vectors."""
    print_section("3. Testing Search")
    
    try:
        results = vector_store.search(
            query="test memory",
            vectors=[0.1] * 1536,
            limit=3
        )
        print(f"✓ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  {i}. ID: {result.id}, Score: {result.score:.4f}")
        return results
    except Exception as e:
        print(f"✗ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_get(vector_store):
    """Test getting a vector."""
    print_section("4. Testing Get")
    
    try:
        result = vector_store.get("test_1")
        if result:
            print(f"✓ Retrieved vector: {result.id}")
            print(f"  Data: {result.payload.get('data', 'N/A')}")
        else:
            print("✗ Vector not found")
        return result
    except Exception as e:
        print(f"✗ Get failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_filter_search(vector_store):
    """Test searching with filters."""
    print_section("5. Testing Filter Search")
    
    try:
        results = vector_store.search(
            query="test",
            vectors=[0.15] * 1536,
            limit=10,
            filters={"user_id": "user1"}
        )
        print(f"✓ Filtered search found {len(results)} results")
        for result in results:
            print(f"  - {result.id}: user_id={result.payload.get('user_id')}")
        return results
    except Exception as e:
        print(f"✗ Filter search failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_update(vector_store):
    """Test updating a vector."""
    print_section("6. Testing Update")
    
    try:
        vector_store.update(
            "test_1",
            vector=[0.5] * 1536,
            payload={"data": "Updated test memory", "updated": True}
        )
        print("✓ Updated vector successfully")
        
        # Verify update
        result = vector_store.get("test_1")
        if result and "Updated" in result.payload.get("data", ""):
            print("✓ Update verified")
        return True
    except Exception as e:
        print(f"✗ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_list(vector_store):
    """Test listing all vectors."""
    print_section("7. Testing List")
    
    try:
        results = vector_store.list(limit=10)
        if results and len(results) > 0:
            print(f"✓ Listed {len(results)} vectors")
            for result in results[:3]:
                print(f"  - {result.id}")
        else:
            print("✗ No vectors found")
        return results
    except Exception as e:
        print(f"✗ List failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_hybrid_search(vector_store):
    """Test hybrid search (vector + text)."""
    print_section("8. Testing Hybrid Search")
    
    try:
        results = vector_store.search(
            query="First test",  # Text query
            vectors=[0.1] * 1536,   # Vector query
            limit=5
        )
        print(f"✓ Hybrid search found {len(results)} results")
        for result in results[:3]:
            data = result.payload.get("data", "N/A") if result.payload else "N/A"
            print(f"  - {result.id}: score={result.score:.4f}, data={data}")
        return results
    except Exception as e:
        print(f"✗ Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_collection_info(vector_store):
    """Test getting collection info."""
    print_section("9. Testing Collection Info")
    
    try:
        info = vector_store.col_info()
        print("✓ Collection info retrieved")
        if CONFIG["collection_name"] in info:
            print(f"  Index: {CONFIG['collection_name']}")
            print(f"  Details: {info[CONFIG['collection_name']]}")
        return info
    except Exception as e:
        print(f"✗ Collection info failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_delete(vector_store):
    """Test deleting a vector."""
    print_section("10. Testing Delete")
    
    try:
        vector_store.delete("test_2")
        print("✓ Deleted vector test_2")
        
        # Verify deletion
        result = vector_store.get("test_2")
        if result is None:
            print("✓ Deletion verified")
        else:
            print("✗ Vector still exists")
        return True
    except Exception as e:
        print(f"✗ Delete failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_integration():
    """Test Memory class integration."""
    print_section("11. Testing Memory Integration")
    
    try:
        from mem0 import Memory
        from mem0.configs.base import MemoryConfig, VectorStoreConfig
        
        config_dict = CONFIG.copy()
        if not config_dict["user"]:
            config_dict.pop("user", None)
        if not config_dict["password"]:
            config_dict.pop("password", None)
        
        config = MemoryConfig(
            vector_store=VectorStoreConfig(
                provider="lindorm_search",
                config=config_dict
            )
        )
        memory = Memory(config)
        
        # Add a memory
        memory.add("Quick test memory from Python script", user_id="quick_test")
        print("✓ Added memory via Memory class")
        
        # Search memories
        results = memory.search("test memory", user_id="quick_test", limit=3)
        print(f"✓ Found {len(results['results'])} memories via Memory class")
        
        # Clean up
        if len(results['results']) > 0:
            memory.delete(results['results'][0]['id'])
            print("✓ Cleaned up test memory")
        
        return True
    except Exception as e:
        print(f"✗ Memory integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup(vector_store):
    """Clean up test data."""
    print_section("12. Cleanup")
    
    try:
        vector_store.delete_col()
        print(f"✓ Deleted test collection: {CONFIG['collection_name']}")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" Lindorm Search - Quick Test")
    print("=" * 60)
    
    # Check if host is configured
    if CONFIG["host"] == "YOUR_LINDORM_HOST":
        print("\n⚠️  WARNING: Lindorm host not configured!")
        print("\nPlease edit this file and set your connection details:")
        print("  1. Open: tests/vector_stores/quick_lindorm_test.py")
        print("  2. Set CONFIG['host'] to your Lindorm host")
        print("  3. Set CONFIG['user'] and CONFIG['password'] if needed")
        print("\nOr use environment variables:")
        print("  LINDORM_HOST=your_host LINDORM_USER=user LINDORM_PASSWORD=pass python quick_lindorm_test.py")
        return
    
    vector_store = None
    
    try:
        # Run tests
        vector_store = test_connection()
        if not vector_store:
            return
        
        test_insert(vector_store)
        test_search(vector_store)
        test_get(vector_store)
        test_filter_search(vector_store)
        test_update(vector_store)
        test_list(vector_store)
        test_hybrid_search(vector_store)
        test_collection_info(vector_store)
        test_delete(vector_store)
        test_memory_integration()
        
    finally:
        # Always cleanup
        if vector_store:
            cleanup(vector_store)
    
    print_section("Test Complete")
    print("All tests finished!")

if __name__ == "__main__":
    main()
