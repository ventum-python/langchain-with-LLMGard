# test_vector_db.py - Vector DB test karne ke liye

import asyncio
from vector_db import VectorStore

async def test_vector_db():
    print("🧪 Testing Pinecone Vector Store...")
    
    try:
        # Step 1: Initialize
        print("\n1️⃣ Initializing Vector Store...")
        store = VectorStore()
        print("✅ Connected to Pinecone!")
        
        # Step 2: Fake vector banayo (768 dimensions)
        print("\n2️⃣ Creating test vector...")
        fake_vector = [0.05] * 768  # 768 numbers
        
        # Step 3: Vector store karo
        print("\n3️⃣ Storing vector...")
        vec_id = await store.add(fake_vector, {
            "source": "test.pdf",
            "text": "This is a test document about AI and machine learning.",
            "chunk_id": 0
        })
        print(f"✅ Stored with ID: {vec_id}")
        
        # Step 4: Search karo
        print("\n4️⃣ Searching similar vectors...")
        results = await store.search(fake_vector, top_k=1)
        
        if results:
            print(f"✅ Found match!")
            print(f"   Score: {results[0]['score']:.4f}")
            print(f"   Text: {results[0]['metadata']['text'][:50]}...")
        else:
            print("⚠️ No matches found")
        
        # Step 5: Stats dekho
        print("\n5️⃣ Getting stats...")
        stats = await store.get_stats()
        print(f"✅ Stats: {stats}")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()



        

# Run karo
asyncio.run(test_vector_db())