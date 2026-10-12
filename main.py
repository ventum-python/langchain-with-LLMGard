"""
main.py - FastAPI Backend (Voice Enabled)
==========================================

New API Added:
POST /api/speak → Text ko AI voice me convert karo
GET  /api/voices → Available voices ki list
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import json
import os
import time
from typing import Optional

# Import our modules
from config import HOST, PORT, logger, ALLOWED_EXTENSIONS
from rag_engine import DocumentProcessor, EmbeddingService, RAGAgent
from pii_guard import guard as pii_guard  # Presidio PII mask + terminal report
from vector_db import VectorStore
from file_scope import resolve_file_scope

# ✅ NEW: ElevenLabs Voice Import
from elevenlabs_voice import generate_voice, get_available_voices, get_user_quota, VOICES


# ============ GLOBAL VARIABLES ============
processor: Optional[DocumentProcessor] = None
embedder: Optional[EmbeddingService] = None
vector_store: Optional[VectorStore] = None
rag_agent: Optional[RAGAgent] = None


# ============ DATABASE SETUP ============
def get_db():
    conn = sqlite3.connect("chatbot.db")
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_legacy_chats(cursor):
    """Old flat chats ko conversations me shift karo, history lose na ho."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chats'"
    )
    if not cursor.fetchone():
        return

    cursor.execute("SELECT COUNT(*) FROM conversations")
    if cursor.fetchone()[0] > 0:
        return

    cursor.execute(
        "SELECT question, answer, confidence, sources, created_at FROM chats ORDER BY id"
    )
    rows = cursor.fetchall()
    for row in rows:
        title = (row["question"] or "Chat")[:50]
        cursor.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, row["created_at"], row["created_at"]),
        )
        conv_id = cursor.lastrowid
        cursor.execute(
            """INSERT INTO messages (conversation_id, role, content, created_at)
               VALUES (?, 'user', ?, ?)""",
            (conv_id, row["question"], row["created_at"]),
        )
        cursor.execute(
            """INSERT INTO messages
               (conversation_id, role, content, confidence, sources, created_at)
               VALUES (?, 'assistant', ?, ?, ?, ?)""",
            (conv_id, row["answer"], row["confidence"], row["sources"], row["created_at"]),
        )


def init_sqlite():
    """SQLite database initialize karo"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            chunks INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            confidence REAL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            voice_name TEXT,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    _migrate_legacy_chats(cursor)
    _ensure_column(cursor, "conversations", "active_file", "TEXT")
    _ensure_column(cursor, "messages", "meta", "TEXT")
    
    conn.commit()
    conn.close()
    logger.info("✅ SQLite database initialized")


def _ensure_column(cursor, table: str, column: str, decl: str):
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if column not in cols:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def list_uploaded_files():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename FROM documents GROUP BY filename ORDER BY MAX(id) DESC"
    )
    files = [r["filename"] for r in cursor.fetchall() if r["filename"]]
    conn.close()
    return files


# ============ FASTAPI APP ============
app = FastAPI(
    title="Smart Chatbot with RAG + Voice",
    description="Upload documents, ask questions, hear AI voice answers",
    version="2.0"
)


# ============ STARTUP EVENT ============
@app.on_event("startup")
async def startup():
    """Server start hone par sab initialize karo"""
    global processor, embedder, vector_store, rag_agent
    
    try:
        logger.info("🚀 Starting application...\n")
        
        logger.info("📊 Initializing SQLite...")
        init_sqlite()
        
        logger.info("📄 Initializing DocumentProcessor...")
        processor = DocumentProcessor()
        
        logger.info("🧠 Initializing EmbeddingService...")
        embedder = EmbeddingService()
        
        logger.info("🗄️ Initializing VectorStore...")
        vector_store = VectorStore()
        
        logger.info("🤖 Initializing RAGAgent...")
        rag_agent = RAGAgent(vector_store, embedder)
        
        # ✅ NEW: Temp folder for voice files
        os.makedirs("temp", exist_ok=True)
        
        logger.info("\n✅ All systems ready!\n")
        logger.info(f"🌐 Server: http://{HOST}:{PORT}")
        logger.info(f"🔊 Voice API: POST http://{HOST}:{PORT}/api/speak")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise


# ============ HOME PAGE ============
@app.get("/", response_class=HTMLResponse)
async def home():
    """HTML UI serve karo"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="UI file not found")


# ============ API: UPLOAD DOCUMENT ============
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Document upload karo"""
    try:
        logger.info(f"📥 Upload request: {file.filename}")
        
        name = (file.filename or "").lower()
        if not any(name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail="Supported: PDF, Word, Excel, PowerPoint, images, TXT, CSV, JSON, HTML",
            )
        
        content = await file.read()
        
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large! Max 10MB")
        
        result = await processor.process_file(file.filename, content)
        chunks = result['chunks']

        safe_chunks = []
        for i, chunk in enumerate(chunks):
            guarded = pii_guard.redact(chunk, stage=f"file_upload:{file.filename}#chunk{i}")
            safe_chunks.append(guarded.redacted)
        
        embeddings = await embedder.embed_batch(safe_chunks)
        
        for i, (chunk, embedding) in enumerate(zip(safe_chunks, embeddings)):
            await vector_store.add(embedding, {
                "source": file.filename,
                "text": chunk,
                "chunk_id": i
            })
        
        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (filename, chunks) VALUES (?, ?)",
            (file.filename, len(chunks))
        )
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"✅ {file.filename} uploaded! {len(chunks)} chunks created."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ API: CONVERSATIONS ============
@app.get("/api/conversations")
async def list_conversations():
    """Saari chats ki history — delete nahi hoti"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, title, created_at, updated_at
               FROM conversations
               ORDER BY datetime(updated_at) DESC, id DESC"""
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return {"conversations": rows}
    except Exception as e:
        logger.error(f"❌ Conversations list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations")
async def create_conversation():
    """Nayi empty chat"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO conversations (title) VALUES (?)", ("New chat",))
        conv_id = cursor.lastrowid
        cursor.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,))
        row = dict(cursor.fetchone())
        conn.commit()
        conn.close()
        return row
    except Exception as e:
        logger.error(f"❌ Create conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    """Sirf ek chat delete karo"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Chat not found")
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
        logger.info(f"🗑️ Deleted conversation {conversation_id}")
        return {"status": "deleted", "id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations")
async def delete_all_conversations():
    """Saari chats DB se delete karo"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        cursor.execute("DELETE FROM chats")
        conn.commit()
        conn.close()
        logger.info("🗑️ All chat history deleted")
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"❌ Clear chats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    """Ek purani chat ke saare messages"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        conv = cursor.fetchone()
        if not conv:
            conn.close()
            raise HTTPException(status_code=404, detail="Chat not found")

        cursor.execute(
            """SELECT id, role, content, confidence, sources, created_at, meta
               FROM messages
               WHERE conversation_id = ?
               ORDER BY id ASC""",
            (conversation_id,),
        )
        messages = []
        for row in cursor.fetchall():
            item = dict(row)
            item["sources"] = [s for s in (item.get("sources") or "").split(",") if s]
            meta = {}
            if item.get("meta"):
                try:
                    meta = json.loads(item["meta"])
                except Exception:
                    meta = {}
            item["pick_file"] = bool(meta.get("pick_file"))
            item["pending_action"] = meta.get("action")
            item["files"] = meta.get("files") or item["sources"]
            messages.append(item)
        conn.close()
        data = dict(conv)
        data["active_file"] = conv["active_file"] if "active_file" in conv.keys() else None
        data["messages"] = messages
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    return {"files": list_uploaded_files()}


# ============ API: ASK QUESTION ============
@app.post("/api/ask")
async def ask_question(request: dict):
    """Question answer karo — current conversation me save hota hai"""
    try:
        question = request.get("question")
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question required!")

        conversation_id = request.get("conversation_id")
        
        logger.info(f"❓ Question: {question}")

        conn = get_db()
        cursor = conn.cursor()

        if conversation_id:
            cursor.execute(
                "SELECT id, title, active_file FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            conv = cursor.fetchone()
            if not conv:
                conn.close()
                raise HTTPException(status_code=404, detail="Chat not found")
        else:
            cursor.execute("INSERT INTO conversations (title) VALUES (?)", ("New chat",))
            conversation_id = cursor.lastrowid
            cursor.execute(
                "SELECT id, title, active_file FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            conv = cursor.fetchone()

        active_file = None
        try:
            active_file = conv["active_file"]
        except Exception:
            active_file = None

        files = list_uploaded_files()
        scope = resolve_file_scope(
            question,
            files,
            requested=request.get("source"),
            active_file=active_file,
        )

        if scope.get("pick"):
            result = {
                "answer": "Kaunsi file ka kaam karun? Neeche file name pe click karo — usi file par chat chalegi.",
                "sources": scope["files"],
                "confidence": 0.0,
                "thinking": "Waiting for file pick",
                "pick_file": True,
                "files": scope["files"],
                "pending_action": scope.get("action") or "summary",
            }
        else:
            result = await rag_agent.answer_question(
                question, top_k=12, source=scope.get("source")
            )
            pin = scope.get("pin")
            if scope.get("clear"):
                cursor.execute(
                    "UPDATE conversations SET active_file = NULL WHERE id = ?",
                    (conversation_id,),
                )
            elif pin:
                cursor.execute(
                    "UPDATE conversations SET active_file = ? WHERE id = ?",
                    (pin, conversation_id),
                )

        sources_str = ",".join(result.get("sources") or [])
        meta = json.dumps({
            "pick_file": bool(result.get("pick_file")),
            "action": result.get("pending_action"),
            "files": result.get("files") or [],
            "active_file": scope.get("source") or scope.get("pin"),
        })

        cursor.execute(
            """INSERT INTO messages (conversation_id, role, content)
               VALUES (?, 'user', ?)""",
            (conversation_id, question),
        )
        cursor.execute(
            """INSERT INTO messages (conversation_id, role, content, confidence, sources, meta)
               VALUES (?, 'assistant', ?, ?, ?, ?)""",
            (conversation_id, result["answer"], result.get("confidence"), sources_str, meta),
        )

        # First user message se title set karo
        if conv["title"] in (None, "", "New chat"):
            title = question.strip()[:50]
            cursor.execute(
                "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, conversation_id),
            )
        else:
            cursor.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,),
            )

        cursor.execute(
            "INSERT INTO chats (question, answer, confidence, sources) VALUES (?, ?, ?, ?)",
            (question, result["answer"], result["confidence"], sources_str),
        )
        conn.commit()
        conn.close()
        
        return {
            "answer": result["answer"],
            "sources": result.get("sources") or [],
            "confidence": f"{(result.get('confidence') or 0):.2%}",
            "thinking": result.get("thinking"),
            "conversation_id": conversation_id,
            "title": question.strip()[:50] if conv["title"] in (None, "", "New chat") else conv["title"],
            "pick_file": bool(result.get("pick_file")),
            "files": result.get("files") or [],
            "pending_action": result.get("pending_action"),
            "active_file": scope.get("source") or scope.get("pin"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW API: TEXT TO SPEECH ============
@app.post("/api/speak")
async def text_to_speech(request: dict):
    """
    Text ko AI voice me convert karo (ElevenLabs)
    
    Request:
    {
        "text": "Answer text here...",
        "voice": "rachel"  // Optional: rachel, adam, bella, etc.
    }
    
    Response:
    {
        "status": "success",
        "audio_url": "/temp/voice_1234567890.mp3",
        "voice_used": "rachel"
    }
    """
    try:
        text = request.get("text", "")
        voice_name = request.get("voice", "rachel")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text required!")
        
        logger.info(f"🔊 Voice request: {text[:50]}... (Voice: {voice_name})")
        
        # Unique filename
        timestamp = int(time.time())
        output_file = f"temp/voice_{timestamp}.mp3"
        
        # Voice generate karo
        result = generate_voice(text, voice_id=voice_name, output_file=output_file)
        
        if result and os.path.exists(result):
            # SQLite me log karo
            conn = sqlite3.connect("chatbot.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO voice_logs (text, voice_name, file_path)
                VALUES (?, ?, ?)
            """, (text[:200], voice_name, result))
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Voice generated: {result}")
            
            return {
                "status": "success",
                "audio_url": f"/temp/{os.path.basename(result)}",
                "voice_used": voice_name,
                "file_size": os.path.getsize(result)
            }
        else:
            raise HTTPException(status_code=500, detail="Voice generation failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Speak error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW API: GET AVAILABLE VOICES ============
@app.get("/api/voices")
async def list_voices():
    """Available AI voices ki list"""
    try:
        voices = []
        for name, voice_id in VOICES.items():
            voices.append({
                "name": name,
                "id": voice_id,
                "preview": f"https://api.elevenlabs.io/v1/voices/{voice_id}/preview"  # Preview URL
            })
        
        return {
            "voices": voices,
            "default": "rachel",
            "total": len(voices)
        }
        
    except Exception as e:
        logger.error(f"❌ Voices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW API: CHECK QUOTA ============
@app.get("/api/quota")
async def check_quota():
    """ElevenLabs quota check karo"""
    try:
        quota = get_user_quota()
        return quota or {"error": "Could not fetch quota"}
    except Exception as e:
        logger.error(f"❌ Quota error: {e}")
        return {"error": str(e)}


# ============ API: STATISTICS ============
@app.get("/api/stats")
async def get_stats():
    """System statistics"""
    try:
        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chats")
        chat_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM voice_logs")
        voice_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conv_count = cursor.fetchone()[0]
        
        conn.close()
        
        vdb_stats = await vector_store.get_stats()
        
        return {
            "documents": doc_count,
            "chats": chat_count,
            "conversations": conv_count,
            "voice_generations": voice_count,
            "vectors": vdb_stats.get('total_vectors', 0)
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"error": str(e)}


# ============ API: CLEAR ALL ============
@app.post("/api/clear")
async def clear_all():
    """Sab kuch delete karo"""
    try:
        logger.info("🗑️ Clearing all data...")
        
        await vector_store.clear()
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        # Chat history persist — conversations/messages delete nahi honge
        conn.commit()
        conn.close()
        
        # Temp files delete karo
        for file in os.listdir("temp"):
            if file.startswith("voice_"):
                os.remove(os.path.join("temp", file))
        
        logger.info("✅ All data cleared")
        return {"status": "cleared"}
        
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ NEW: Temp folder for voice files (create before mounting!)
os.makedirs("temp", exist_ok=True)

# ✅ NEW: Static files for voice (IMPORTANT!)
app.mount("/temp", StaticFiles(directory="temp"), name="temp")


# ============ RUN SERVER ============
if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("🚀 SMART CHATBOT WITH RAG + VOICE")
    logger.info("=" * 60)
    logger.info(f"📄 UI: http://localhost:{PORT}")
    logger.info(f"📚 API Docs: http://localhost:{PORT}/docs")
    logger.info(f"🎤 Voice API: POST http://localhost:{PORT}/api/speak")
    logger.info(f"🎵 Voices: GET http://localhost:{PORT}/api/voices")
    logger.info("=" * 60 + "\n")
    
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")