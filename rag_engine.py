"""
rag_engine.py - Complete RAG Pipeline (GROQ FIX)
==================================================

FIX: Gemini rate limit exceeded
SOLUTION: Use Groq API for LLM (free tier: 20 req/min)
"""

import logging
import re
import time
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import GEMINI_API_KEY, GROQ_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, logger
from pii_guard import guard as pii_guard
from file_extractors import extract_text

# Google API for embeddings (ye still kaam kar raha hai)
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

# Groq API for LLM (fast, no rate limits!)
import groq

# ============ GROQ CLIENT ============
groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ============ PART 1: DOCUMENT PROCESSOR ============
class DocumentProcessor:
    """Documents ko process karti hai"""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"📄 DocumentProcessor ready (chunk_size={CHUNK_SIZE})")
    
    def split_to_chunks(self, text: str) -> List[str]:
        """Text ko chunks me todo"""
        try:
            chunks = self.splitter.split_text(text)
            avg_size = len(text) / len(chunks) if chunks else 0
            logger.info(f"✂️ {len(chunks)} chunks (avg {avg_size:.0f} chars each)")
            return chunks
        except Exception as e:
            logger.error(f"❌ Split error: {e}")
            raise
    
    async def process_file(self, filename: str, file_content: bytes) -> Dict:
        """Complete processing pipeline"""
        try:
            logger.info(f"📥 Processing: {filename}")
            
            text = extract_text(filename, file_content)
            
            chunks = self.split_to_chunks(text)
            
            result = {
                'chunks': chunks,
                'filename': filename,
                'file_size': len(file_content),
                'chunk_count': len(chunks),
                'preview': text[:200]
            }
            
            logger.info(f"✅ Done: {filename} → {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            raise


# ============ PART 2: EMBEDDING SERVICE ============
class EmbeddingService:
    """Google API se embeddings banati hai"""
    
    def __init__(self):
        self.model_name = "models/gemini-embedding-001"
        self.cache = {}
        logger.info(f"🧠 EmbeddingService ready (Google API: {self.model_name})")
    
    async def embed_text(self, text: str) -> List[float]:
        """Single text ko embedding me convert karo"""
        try:
            if text in self.cache:
                return self.cache[text]
            
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            self.cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Question ko embedding me convert karo"""
        try:
            if text in self.cache:
                return self.cache[text]
            
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            
            embedding = result['embedding']
            self.cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Query embedding error: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Multiple texts ko ek saath embed karo"""
        try:
            logger.info(f"⚡ Batch embedding {len(texts)} texts...")
            embeddings = []
            
            for i, text in enumerate(texts):
                embedding = await self.embed_text(text)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"   Progress: {i + 1}/{len(texts)}")
            
            logger.info(f"✅ Batch complete: {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch error: {e}")
            raise


_PII_TAG = re.compile(
    r"<(PERSON|IN_PAN|IN_GSTIN|IN_AADHAAR|IN_IFSC|IN_PHONE|IN_BANK_ACCOUNT|"
    r"EMAIL_ADDRESS|PHONE_NUMBER|CREDIT_CARD|IBAN_CODE|ORGANIZATION|LOCATION)>",
    re.I,
)


def clean_llm_text(text: str, allow_table: bool = False) -> str:
    """HTML / PII tags hatao. Table tabhi rakho jab user ne maanga ho."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = _PII_TAG.sub("", text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not allow_table:
        text = re.sub(r"(?m)^\s*\|?\s*-{2,}\s*(\|\s*-{2,}\s*)+\|?\s*$", "", text)
        text = re.sub(r"(?m)^\s*\|", "", text)
        text = re.sub(r"\|", " — ", text)
    return text.strip()


# ============ PART 3: RAG AGENT (GROQ FIX!) ============
class RAGAgent:
    """
    RAG Pipeline - Retrieve + Augment + Generate
    
    FIX: Use Groq for LLM instead of Gemini (rate limit issue)
    """
    
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Groq retired llama-3.1-8b-instant (Aug 16, 2026). Official replacement:
        self.groq_model = "openai/gpt-oss-20b"
        
        # Fallback to Gemini if Groq not available
        self.gemini_model = "gemini-2.0-flash"
        
        logger.info(f"🤖 RAGAgent initialized (Groq: {self.groq_model})")
    
    async def answer_question(self, question: str, top_k: int = 12, source: str = None) -> Dict:
        """Complete RAG pipeline"""
        try:
            logger.info(f"❓ Question: {question}")
            
            # Step 1: RETRIEVE
            logger.info("1️⃣ RETRIEVING documents...")
            query_embedding = await self.embedding_service.embed_query(question)

            # Resume/small docs: pull enough chunks so sections like Projects are not missed
            stats = await self.vector_store.get_stats()
            total_vectors = stats.get("total_vectors") or 0
            if total_vectors:
                top_k = max(top_k, min(total_vectors, 15))
            
            if source:
                logger.info(f"🎯 File scope: {source}")
            retrieved = await self.vector_store.search(
                query_embedding, top_k=top_k, source=source
            )
            
            if not retrieved:
                logger.warning("⚠️ No documents found")
                return {
                    "answer": "Sorry, mujhe is question ka answer documents me nahi mila. Pehle koi document upload karo!",
                    "sources": [],
                    "confidence": 0.0,
                    "thinking": "No documents in database"
                }
            
            logger.info(f"✅ Retrieved {len(retrieved)} documents")
            
            # Step 2: AUGMENT
            logger.info("2️⃣ AUGMENTING context...")
            
            context_text = ""
            sources = []
            
            for i, match in enumerate(retrieved):
                source = match['metadata'].get('source', 'Unknown')
                chunk_text = match['metadata'].get('text', '')
                score = match['score']
                
                sources.append(source)
                context_text += f"\n[Document {i+1} - {source}]\n{chunk_text}\n"
            
            logger.info(f"✅ Context prepared ({len(context_text)} chars)")

            safe_docs = pii_guard.redact(context_text, stage="before_llm.documents")
            safe_q = pii_guard.redact(question, stage="before_llm.question")
            
            # Step 3: GENERATE (GROQ!)
            logger.info("3️⃣ GENERATING answer with Groq (PII already stripped)...")
            
            wants_visual = bool(
                re.search(r"\b(table|chart|graph|tabular|spreadsheet)\b", question, re.I)
            )
            style = (
                "The user asked for a table or chart. Use a simple markdown table with | columns. "
                "Do not put HTML inside cells."
                if wants_visual
                else
                "Write like ChatGPT: short paragraphs and bullet lists only. "
                "Do NOT use markdown tables, pipe | layouts, or HTML."
            )

            scope_rule = (
                f"Answer ONLY from this one file: {source}. Ignore any other document."
                if source
                else "Use the provided documents. If several files appear, say so clearly."
            )

            prompt = f"""You are a helpful document assistant. Use the text below to answer.

Rules:
- {scope_rule}
- Combine facts from every section (skills, experience, projects, education, companies).
- {style}
- Never output HTML tags such as <br>, <p>, <div>, <table>.
- Never output placeholders like <PERSON>, <IN_PAN>, <EMAIL_ADDRESS>. Skip that secret or say "redacted".
- If the question has typos or is informal, interpret the intended meaning.
- If asked how many projects/skills/jobs, count and list what appears in the text.
- Only say you lack information if that topic is completely missing from the text.

Documents:
{safe_docs.redacted}

Question: {safe_q.redacted}

Answer:"""
            
            # ✅ GROQ API CALL
            if groq_client:
                response = groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful resume assistant. Answer in plain ChatGPT-style text. No HTML tags. No markdown tables unless the user asks for a table or chart."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                answer_text = clean_llm_text(response.choices[0].message.content, wants_visual)
            else:
                # Fallback to Gemini
                logger.warning("⚠️ Groq not available, using Gemini...")
                time.sleep(5)  # Rate limit
                model = genai.GenerativeModel(self.gemini_model)
                response = model.generate_content(prompt)
                raw = response.text if hasattr(response, 'text') else str(response)
                answer_text = clean_llm_text(raw, wants_visual)
            
            logger.info(f"✅ Answer generated")
            
            # Confidence calculate karo
            if retrieved:
                confidence = sum(m['score'] for m in retrieved) / len(retrieved)
            else:
                confidence = 0.0
            
            result = {
                "answer": answer_text,
                "sources": list(set(sources)),
                "confidence": confidence,
                "thinking": f"Retrieved {len(retrieved)} documents with avg similarity {confidence:.2%}"
            }
            
            logger.info(f"✅ RAG Complete! Confidence: {confidence:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG Error: {e}")
            return {
                "answer": f"Error occurred: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "thinking": f"Error: {e}"
            }