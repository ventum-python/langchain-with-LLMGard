"""
elevenlabs_voice.py - ElevenLabs AI Voice Integration
======================================================

Ye file kya karti hai?
1. Text ko natural AI voice me convert karti hai
2. MP3 file generate karti hai
3. Multiple voices support karti hai

API: https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
"""

import requests
import os
from config import logger

# ============ CONFIGURATION ============
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_your_key_here")

# Available Voices (Best ones)
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",      # Female - Warm, natural
    "adam": "pNInz6obpgDQGcFmaJgB",         # Male - Deep, professional
    "bella": "EXAVITQu4vr4xnSDxMaL",        # Female - Soft, gentle
    "antoni": "ErXwobaYiN019PkySvjV",       # Male - Energetic
    "elli": "MF3mGyEYCl7XYWbV9V6O",         # Female - Young, bright
    "josh": "TxGEqnHWrfWFTfGW9XjX",         # Male - Calm, soothing
    "arnold": "VR6AewLTigWG4xSOukaG",       # Male - News anchor style
    "domi": "AZnzlk1XvdvUeBnXmlld"          # Female - Strong, confident
}

DEFAULT_VOICE = "rachel"  # Sabse natural voice


def generate_voice(text: str, voice_id: str = None, output_file: str = "temp/voice.mp3") -> str:
    """
    Text ko AI voice me convert karo (ElevenLabs)
    
    Parameters:
    -----------
    text : str
        Jo bolwana hai (answer text)
    
    voice_id : str
        Kaunsi voice use karni hai (default: rachel)
    
    output_file : str
        MP3 file ka path
    
    Returns:
    --------
    str : Generated MP3 file ka path, ya None agar fail ho
    
    Example:
    --------
    file_path = generate_voice("Hello, this is AI speaking!")
    if file_path:
        print(f"Voice saved: {file_path}")
    """
    
    try:
        # Voice ID select karo
        if not voice_id:
            voice_id = VOICES[DEFAULT_VOICE]
        elif voice_id in VOICES:
            voice_id = VOICES[voice_id]
        
        # API URL
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        # Headers
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        # Request body
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,           # 0-1, kam = more expressive
                "similarity_boost": 0.75,    # 0-1, zyada = more like original voice
                "style": 0.0,               # -1 to 1, voice style
                "use_speaker_boost": True   # Better quality
            }
        }
        
        logger.info(f"🔊 Generating voice for: {text[:50]}...")
        
        # API call
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        # Check response
        if response.status_code == 200:
            # Temp folder banao agar nahi hai
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # MP3 save karo
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_file)
            logger.info(f"✅ Voice generated: {output_file} ({file_size} bytes)")
            
            return output_file
            
        elif response.status_code == 401:
            logger.error("❌ Invalid API key!")
            return None
            
        elif response.status_code == 429:
            logger.error("❌ Rate limit exceeded! Free tier: 10K chars/month")
            return None
            
        else:
            logger.error(f"❌ ElevenLabs error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error("❌ Request timeout!")
        return None
        
    except Exception as e:
        logger.error(f"❌ Voice generation error: {e}")
        return None


def get_available_voices() -> list:
    """
    ElevenLabs se available voices ki list lo
    
    Returns:
    --------
    list : [{"name": "Rachel", "id": "21m00Tcm4TlvDq8ikWAM", "category": "premade"}, ...]
    """
    try:
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            voices = response.json().get("voices", [])
            logger.info(f"✅ Found {len(voices)} voices")
            return voices
        else:
            logger.error(f"❌ Error fetching voices: {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return []


def get_user_quota() -> dict:
    """
    Check remaining quota (free tier)
    
    Returns:
    --------
    dict : {"character_count": 5000, "character_limit": 10000, ...}
    """
    try:
        url = "https://api.elevenlabs.io/v1/user/subscription"
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            used = data.get("character_count", 0)
            limit = data.get("character_limit", 10000)
            remaining = limit - used
            
            logger.info(f"📊 Quota: {used}/{limit} used, {remaining} remaining")
            
            return {
                "used": used,
                "limit": limit,
                "remaining": remaining,
                "percentage": (used / limit) * 100 if limit > 0 else 0
            }
        else:
            return {}
            
    except Exception as e:
        logger.error(f"❌ Quota check error: {e}")
        return {}


# ============ TEST ============
if __name__ == "__main__":
    print("🧪 Testing ElevenLabs Voice...")
    
    # Test voice generation
    test_text = "Hello! I am your AI assistant. I can answer questions about your documents."
    
    result = generate_voice(test_text, output_file="test_voice.mp3")
    
    if result:
        print(f"✅ Success! File: {result}")
        print(f"📁 Size: {os.path.getsize(result)} bytes")
    else:
        print("❌ Failed!")
    
    # Check quota
    quota = get_user_quota()
    if quota:
        print(f"\n📊 Quota: {quota['used']}/{quota['limit']} ({quota['percentage']:.1f}%)")