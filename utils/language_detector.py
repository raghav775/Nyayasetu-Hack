"""
Detects whether user input is English, Hindi, or Hinglish.
Used to instruct the LLM to respond in the same language.
"""

from langdetect import detect, DetectorFactory

# Make detection deterministic
DetectorFactory.seed = 0

# Common Hinglish markers — Hindi words written in Latin script
HINGLISH_MARKERS = {
    "kya", "hai", "nahi", "mujhe", "kaise", "karo", "karein",
    "batao", "chahiye", "kaisa", "kyun", "toh", "aur", "lekin",
    "matlab", "samjho", "bolo", "lagta", "wala", "wali", "hoga",
    "tha", "thi", "the", "mera", "meri", "mere", "aapka", "unka",
    "yeh", "woh", "iska", "uska", "kuch", "sab", "sirf", "abhi",
    "pehle", "baad", "zaroor", "bilkul", "theek", "accha", "sahi",
}


def detect_language(text: str) -> str:
    """
    Returns one of: 'hindi', 'hinglish', 'english'

    Strategy:
    - If langdetect says 'hi' → Hindi (Devanagari script)
    - If any Hinglish markers appear in a non-Hindi detected text → Hinglish
    - Otherwise → English
    """
    if not text or len(text.strip()) < 3:
        return "english"

    try:
        detected = detect(text)
    except Exception:
        return "english"

    # Pure Hindi in Devanagari
    if detected == "hi":
        return "hindi"

    # Check for Hinglish (Hindi words in Latin script mixed with English)
    lower_tokens = set(text.lower().split())
    hinglish_hits = lower_tokens.intersection(HINGLISH_MARKERS)

    if len(hinglish_hits) >= 1:
        return "hinglish"

    return "english"


def get_language_instruction(language: str) -> str:
    """
    Returns a system-level instruction to append to prompts
    so the LLM responds in the correct language.
    """
    instructions = {
        "hindi": (
            "उपयोगकर्ता ने हिंदी में प्रश्न पूछा है। "
            "कृपया अपना पूरा उत्तर हिंदी में दें। "
            "कानूनी शब्दावली के लिए आप अंग्रेजी के शब्द रख सकते हैं (जैसे 'FIR', 'IPC', 'affidavit')।"
        ),
        "hinglish": (
            "The user has written in Hinglish (a mix of Hindi and English). "
            "Please respond in Hinglish — natural conversational Hindi mixed with English, "
            "using Latin script throughout. Keep legal terms in English (e.g. FIR, bail, affidavit, IPC)."
        ),
        "english": "",
    }
    return instructions.get(language, "")
