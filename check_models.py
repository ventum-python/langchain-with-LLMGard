# check_models.py - Available models check karo

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("🔍 Available Gemini Models:\n")
for m in genai.list_models():
    print(f"📌 Name: {m.name}")
    print(f"   Supported: {m.supported_generation_methods}")
    print()