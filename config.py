"""
config.py - Project ki Settings (Sabse Pehle ye samjho!)
=========================================================

Ye file kya karti hai?
1. .env file se secrets padhti hai
2. Logging setup karti hai
3. Sab constants ek jagah rakhti hai

Kyun zaroori hai?
→ Agar API key change karni hai, to sirf .env file me change karo
→ Code me kahin bhi hardcoded values nahi honge
→ Production me asani se manage ho sakta hai
"""

import os
from dotenv import load_dotenv
import logging

# ============ .env FILE LOAD KARO ============
# load_dotenv() → .env file ko padhta hai aur variables ko 
# environment variables me load karta hai

load_dotenv()

# ============ LOGGING SETUP ============
# Logging kya hai? → Console pe messages print karna
# Levels: DEBUG < INFO < WARNING < ERROR < CRITICAL

logging.basicConfig(
    level=logging.INFO,  # INFO aur usse upar ke messages dikhenge
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    # %(asctime)s → Time
    # %(name)s → Logger ka naam
    # %(levelname)s → INFO/ERROR/WARNING
    # %(message)s → Actual message
)

# logger object banaya - isse hum logging karenge
logger = logging.getLogger("SmartChatbot")

# ============ API KEYS ============
# os.getenv("KEY_NAME", "default_value") 
# → Environment variable se value padhta hai
# → Agar nahi milta to default value deta hai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# API key check - Agar nahi hai to error!
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found! .env file check karo!")

logger.info("✅ Gemini API Key loaded successfully")

# ============ DATABASE SETTINGS ============
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")
logger.info(f"✅ Database: {DATABASE_URL}")

# ============ LANGCHAIN CHUNK SETTINGS ============
"""
Chunking kya hai?
→ Bade documents ko chote pieces me todna
→ Jaise 5000 words ki PDF hai, to 500-500 words ke chunks banayein

CHUNK_SIZE = 500
→ Har chunk 500 characters ka hoga

CHUNK_OVERLAP = 100  
→ Har chunk ke beech 100 characters overlap hoga
→ Isse context break nahi hota, continuity rehti hai

Example:
Chunk 1: "Machine learning is a... [500 chars]"
Chunk 2: "is a subset of AI... [500 chars]"  
         ↑ ye 100 chars Chunk 1 se overlap hain
"""

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

logger.info(f"✅ Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY not found! Groq features disabled.")

logger.info("✅ Config loaded")

# ============ SERVER SETTINGS ============
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logger.info(f"✅ Server: http://{HOST}:{PORT}")

# ============ CONSTANTS ============
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB = 10 * 1024 KB * 1024 bytes
ALLOWED_EXTENSIONS = [
    ".pdf", ".txt", ".md", ".csv", ".json", ".html", ".htm", ".rtf",
    ".docx", ".doc",
    ".xlsx", ".xls", ".xlsm",
    ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif",
]

logger.info("=" * 50)
logger.info("🚀 Smart Chatbot Config Loaded!")
logger.info("=" * 50)