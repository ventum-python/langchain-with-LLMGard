"""
vector_db.py - Pinecone Vector Database (FIXED)
================================================

FIX: Auto-delete + recreate index if wrong dimension
"""

import logging
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from config import logger

PINECONE_API_KEY = "pcsk_3rmG3d_HYgJHYAPgDBwttgntkbKF69DevbjvjH5zErb1VVqKoDKtZwbJkaRozyf1gGzLa6"

INDEX_NAME = "smart-chatbot"
DIMENSION = 3072  # ✅ Gemini embeddings 3072 dimensions
METRIC = "cosine"


class VectorStore:
    """Pinecone Vector Store"""
    
    def __init__(self):
        logger.info("🗄️ Initializing Pinecone Vector Store...")
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        logger.info("✅ Connected to Pinecone")
        
        self._ensure_index()
        self.index = self.pc.Index(INDEX_NAME)
        logger.info(f"✅ Index '{INDEX_NAME}' ready")
    
    def _ensure_index(self):
        """Index check + auto-fix dimension"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        logger.info(f"📋 Existing indexes: {existing_indexes}")
        
        if INDEX_NAME in existing_indexes:
            try:
                # Dimension check karo
                temp_index = self.pc.Index(INDEX_NAME)
                stats = temp_index.describe_index_stats()
                
                # Agar dimension galat hai to delete
                if hasattr(stats, 'dimension') and stats.dimension != DIMENSION:
                    logger.warning(f"⚠️ Wrong dimension! Expected {DIMENSION}, got {stats.dimension}")
                    logger.info(f"🗑️ Deleting old index...")
                    self.pc.delete_index(INDEX_NAME)
                    logger.info(f"✅ Old index deleted")
                else:
                    logger.info(f"✅ Index OK (dimension: {DIMENSION})")
                    return
                    
            except Exception as e:
                logger.warning(f"⚠️ Error: {e}")
                self.pc.delete_index(INDEX_NAME)
        
        # Naya index create karo
        logger.info(f"🔨 Creating index: {INDEX_NAME} (dim={DIMENSION})")
        self.pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info(f"✅ Index created")
    
    async def add(self, vector: List[float], metadata: Dict) -> str:
        """Vector store karo"""
        try:
            vec_id = f"vec_{metadata.get('source', 'unknown')}_{metadata.get('chunk_id', 0)}"
            
            self.index.upsert(
                vectors=[{
                    "id": vec_id,
                    "values": vector,
                    "metadata": metadata
                }],
                namespace="default"
            )
            
            logger.info(f"✅ Vector stored: {vec_id}")
            return vec_id
            
        except Exception as e:
            logger.error(f"❌ Error storing vector: {e}")
            raise
    
    async def search(self, query_vector: List[float], top_k: int = 3, source: str = None) -> List[Dict]:
        """Similar vectors search karo — source diya ho to sirf usi file se."""
        try:
            if not self.index:
                return []

            query = {
                "vector": query_vector,
                "top_k": top_k,
                "namespace": "default",
                "include_metadata": True,
            }
            if source:
                query["filter"] = {"source": {"$eq": source}}
            
            results = self.index.query(**query)
            
            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            logger.info(f"🔍 Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    async def get_stats(self) -> Dict:
        """Stats return karo"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": DIMENSION,
                "index_name": INDEX_NAME
            }
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {}
    
    async def clear(self):
        """Sab delete karo"""
        try:
            self.index.delete(delete_all=True, namespace="default")
            logger.info("✅ Vectors cleared")
        except Exception as e:
            logger.error(f"❌ Clear error: {e}")
            raise