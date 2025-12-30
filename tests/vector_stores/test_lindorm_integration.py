"""
Lindorm Search Integration Tests

These tests require a real Lindorm Search instance. Set the following environment variables:
- LINDORM_HOST: Lindorm Search host address
- LINDORM_PORT: Lindorm Search port (default: 30070)
- LINDORM_USER: Lindorm Search username (default: root)
- LINDORM_PASSWORD: Lindorm Search password
"""

import os
import unittest
from datetime import datetime
from mem0.vector_stores.lindorm_search import LindormSearch


class TestLindormSearchIntegration(unittest.TestCase):
    """Integration tests for Lindorm Search with real connection."""

    @classmethod
    def setUpClass(cls):
        """Set up test connection using environment variables."""
        cls.host = os.getenv("LINDORM_HOST", "localhost")
        cls.port = int(os.getenv("LINDORM_PORT", 30070))
        cls.user = os.getenv("LINDORM_USER", "")
        cls.password = os.getenv("LINDORM_PASSWORD", "")
        cls.collection_name = f"test_mem0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Skip tests if host not provided
        if not cls.host or cls.host == "localhost":
            raise unittest.SkipTest("LINDORM_HOST environment variable not set")

        cls.vector_store = None

    def setUp(self):
        """Create a new collection for each test."""
        if self.__class__.vector_store is None:
            try:
                config = {
                    "host": self.host,
                    "port": self.port,
                    "collection_name": self.collection_name,
                    "embedding_model_dims": 1536,
                    "distance_method": "cosinesimil"
                }
                
                if self.user:
                    config["user"] = self.user
                if self.password:
                    config["password"] = self.password

                self.__class__.vector_store = LindormSearch(**config)
            except Exception as e:
                self.skipTest(f"Failed to connect to Lindorm: {e}")

    def test_01_create_collection(self):
        """Test creating a new collection."""
        # Collection should be created in __init__
        collections = self.vector_store.list_cols()
        self.assertIn(self.collection_name, collections)
        print(f"✓ Collection {self.collection_name} created successfully")

    def test_02_insert_vectors(self):
        """Test inserting vectors into the collection."""
        vectors = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536
        ]
        payloads = [
            {
                "hash": "hash1",
                "data": "Test memory 1",
                "user_id": "user1",
                "created_at": datetime.now().isoformat()
            },
            {
                "hash": "hash2",
                "data": "Test memory 2",
                "user_id": "user1",
                "created_at": datetime.now().isoformat()
            },
            {
                "hash": "hash3",
                "data": "Test memory 3",
                "user_id": "user2",
                "created_at": datetime.now().isoformat()
            }
        ]
        ids = ["test_id_1", "test_id_2", "test_id_3"]

        results = self.vector_store.insert(vectors=vectors, payloads=payloads, ids=ids)
        self.assertEqual(len(results), 3)
        print(f"✓ Inserted {len(vectors)} vectors successfully")

    def test_03_search_vectors(self):
        """Test searching for similar vectors."""
        query_vector = [0.1] * 1536
        results = self.vector_store.search(
            query="Test memory",
            vectors=query_vector,
            limit=3
        )

        self.assertGreater(len(results), 0)
        self.assertGreaterEqual(results[0].score, 0)
        print(f"✓ Found {len(results)} results")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: id={result.id}, score={result.score:.4f}")

    def test_04_search_with_filters(self):
        """Test searching with user_id filter."""
        query_vector = [0.15] * 1536
        results = self.vector_store.search(
            query="Test memory",
            vectors=query_vector,
            limit=10,
            filters={"user_id": "user1"}
        )

        print(f"✓ Search with filters found {len(results)} results")
        for result in results:
            if result.payload and "user_id" in result.payload:
                self.assertEqual(result.payload["user_id"], "user1")

    def test_05_get_vector(self):
        """Test retrieving a specific vector by ID."""
        result = self.vector_store.get("test_id_1")

        self.assertIsNotNone(result)
        self.assertEqual(result.id, "test_id_1")
        self.assertIsNotNone(result.payload)
        print(f"✓ Retrieved vector: {result.id}")
        print(f"  Payload: {result.payload}")

    def test_06_update_vector(self):
        """Test updating a vector."""
        new_vector = [0.5] * 1536
        new_payload = {
            "data": "Updated test memory",
            "updated_at": datetime.now().isoformat()
        }

        self.vector_store.update("test_id_1", vector=new_vector, payload=new_payload)

        # Verify update
        result = self.vector_store.get("test_id_1")
        self.assertIsNotNone(result)
        self.assertIn("Updated", result.payload.get("data", ""))
        print(f"✓ Updated vector: {result.id}")

    def test_07_list_vectors(self):
        """Test listing all vectors."""
        results = self.vector_store.list(limit=10)

        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        print(f"✓ Listed {len(results)} vectors")

    def test_08_hybrid_search(self):
        """Test hybrid search (vector + text)."""
        query_vector = [0.1] * 1536
        results = self.vector_store.search(
            query="Test memory 1",  # Text query
            vectors=query_vector,   # Vector query
            limit=5
        )

        self.assertGreater(len(results), 0)
        print(f"✓ Hybrid search found {len(results)} results")
        for result in results[:3]:
            print(f"  - {result.id}: score={result.score:.4f}, data={result.payload.get('data', 'N/A')}")

    def test_09_collection_info(self):
        """Test getting collection information."""
        info = self.vector_store.col_info()

        self.assertIsNotNone(info)
        self.assertIsInstance(info, dict)
        print("✓ Collection info retrieved")
        if self.collection_name in info:
            print(f"  Info: {info[self.collection_name]}")

    def test_10_delete_vector(self):
        """Test deleting a vector."""
        self.vector_store.delete("test_id_2")

        # Verify deletion
        result = self.vector_store.get("test_id_2")
        self.assertIsNone(result)
        print("✓ Deleted vector successfully")

    def test_11_delete_collection(self):
        """Test deleting a collection."""
        # Create a temporary collection
        temp_collection = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_store = LindormSearch(
            host=self.host,
            port=self.port,
            user=self.user if self.user else None,
            password=self.password if self.password else None,
            collection_name=temp_collection,
            embedding_model_dims=1536
        )
        
        # Verify it exists
        collections = temp_store.list_cols()
        self.assertIn(temp_collection, collections)
        
        # Delete it
        temp_store.delete_col()
        
        # Verify deletion
        collections = self.vector_store.list_cols()
        self.assertNotIn(temp_collection, collections)
        print(f"✓ Deleted collection: {temp_collection}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test collections."""
        if cls.vector_store:
            try:
                # Delete test collection
                cls.vector_store.delete_col()
                print(f"\n✓ Cleaned up test collection: {cls.collection_name}")
            except Exception as e:
                print(f"\nWarning: Failed to clean up collection: {e}")


class TestLindormSearchWithMemory(unittest.TestCase):
    """Integration tests for Lindorm Search with Memory class."""

    @classmethod
    def setUpClass(cls):
        """Set up test connection."""
        cls.host = os.getenv("LINDORM_HOST", "localhost")
        cls.port = int(os.getenv("LINDORM_PORT", 30070))
        cls.user = os.getenv("LINDORM_USER", "")
        cls.password = os.getenv("LINDORM_PASSWORD", "")

        if not cls.host or cls.host == "localhost":
            raise unittest.SkipTest("LINDORM_HOST environment variable not set")

    def test_memory_with_lindorm(self):
        """Test Memory class integration with Lindorm Search."""
        from mem0 import Memory
        from mem0.configs.base import MemoryConfig, VectorStoreConfig

        config = {
            "host": self.host,
            "port": self.port,
            "collection_name": f"test_mem0_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "embedding_model_dims": 1536
        }
        
        if self.user:
            config["user"] = self.user
        if self.password:
            config["password"] = self.password

        memory_config = MemoryConfig(
            vector_store=VectorStoreConfig(
                provider="lindorm_search",
                config=config
            )
        )

        memory = Memory(memory_config)

        # Test adding memory
        result = memory.add("I love programming in Python", user_id="test_user")
        self.assertIsNotNone(result)
        print(f"✓ Added memory: {result}")

        # Test searching memory
        results = memory.search("programming", user_id="test_user", limit=3)
        self.assertGreater(len(results["results"]), 0)
        print(f"✓ Found {len(results['results'])} memories")

        # Test getting memory
        memory_id = results["results"][0]["id"]
        memory_item = memory.get(memory_id)
        self.assertIsNotNone(memory_item)
        print(f"✓ Retrieved memory: {memory_item['memory']}")

        # Test updating memory
        updated = memory.update(memory_id, "I love coding in Python")
        self.assertIsNotNone(updated)
        print("✓ Updated memory")

        # Test deleting memory
        memory.delete(memory_id)
        print("✓ Deleted memory")


def run_tests():
    """Run tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestLindormSearchIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestLindormSearchWithMemory))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
