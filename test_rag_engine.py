# test_rag_complete.py - Complete RAG Pipeline test

import asyncio
from rag_engine import DocumentProcessor, EmbeddingService, RAGAgent
from vector_db import VectorStore

async def test_complete_rag():
    print("🧪 Testing Complete RAG Pipeline...")
    
    try:
        # Step 1: Initialize
        print("\n1️⃣ Initializing all components...")
        processor = DocumentProcessor()
        embedder = EmbeddingService()
        vector_store = VectorStore()
        agent = RAGAgent(vector_store, embedder)
        print("✅ All components ready!")
        
        # Step 2: Sample document create karo
        print("\n2️⃣ Creating sample document...")
        sample_text = """
        Machine learning is a subset of artificial intelligence.
        It enables computers to learn from data without explicit programming.
        Machine learning algorithms build models based on sample data.
        
        Deep learning is a type of machine learning.
        It uses neural networks with many layers.
        Deep learning is used in image recognition and NLP.
        
        Natural language processing helps computers understand human language.
        It is used in chatbots, translation, and sentiment analysis.
        NLP combines machine learning with linguistics.
        """
        
        # Step 3: Process karo
        print("\n3️⃣ Processing document...")
        chunks = processor.split_to_chunks(sample_text)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 4: Embeddings banao
        print("\n4️⃣ Creating embeddings...")
        embeddings = await embedder.embed_batch(chunks)
        print(f"✅ Created {len(embeddings)} embeddings")
        
        # Step 5: Pinecone me store karo
        print("\n5️⃣ Storing in Pinecone...")
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await vector_store.add(embedding, {
                "source": "sample_doc.txt",
                "text": chunk,
                "chunk_id": i
            })
        print("✅ Stored in Pinecone!")
        
        # Step 6: Question poocho
        print("\n6️⃣ Asking question...")
        question = "What is machine learning?"
        result = await agent.answer_question(question, top_k=2)
        
        print(f"\n🎯 QUESTION: {question}")
        print(f"📝 ANSWER: {result['answer']}")
        print(f"📎 SOURCES: {result['sources']}")
        print(f"🎯 CONFIDENCE: {result['confidence']:.2%}")
        print(f"💭 THINKING: {result['thinking']}")
        
        # Step 7: Stats check karo
        print("\n7️⃣ Checking stats...")
        stats = await vector_store.get_stats()
        print(f"✅ Total vectors: {stats.get('total_vectors', 0)}")
        
        print("\n🎉 Complete RAG Pipeline Test Passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_complete_rag())