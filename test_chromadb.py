import chromadb
import os

print("Testing ChromaDB...")
print("-" * 50)

# 1. Create client (this creates the database automatically)
db_path = os.path.join(os.getcwd(), "data", "chroma_db")
print(f"1. Creating database at: {db_path}")
client = chromadb.PersistentClient(path=db_path)
print("   ✅ Database created!")

# 2. Create collection (like a table)
print("\n2. Creating collection...")
collection = client.get_or_create_collection(name="test_collection")
print("   ✅ Collection created!")

# 3. Add some test data
print("\n3. Adding test data...")
collection.add(
    ids=["test1", "test2"],
    documents=["This is a test document about Python.", "This is about machine learning."],
    embeddings=[[0.1] * 384, [0.2] * 384],  # Dummy embeddings (384 dimensions)
    metadatas=[{"type": "test"}, {"type": "test"}]
)
print("   ✅ Data added!")

# 4. Query the data
print("\n4. Querying data...")
results = collection.query(
    query_embeddings=[[0.1] * 384],
    n_results=1
)
print(f"   ✅ Found {len(results['ids'][0])} result(s)")
print(f"   Document: {results['documents'][0][0]}")

# 5. Check if data persists
print("\n5. Testing persistence...")
new_client = chromadb.PersistentClient(path=db_path)
new_collection = new_client.get_collection(name="test_collection")
count = new_collection.count()
print(f"   ✅ Collection has {count} items (data persisted!)")

print("\n" + "=" * 50)
print("✅ ChromaDB is working correctly!")
print("=" * 50)