# CreatorCore Story Agent Configuration

# --- API and Model Settings ---

# Default models to use for different tasks
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_PRO_MODEL = "gemini-2.5-pro"

# Default temperature setting for creative tasks
DEFAULT_TEMPERATURE = 0.7

# --- Application Metadata ---
APP_TITLE = "CreatorCore Story Agent API"
APP_DESCRIPTION = "Modular backend for AI Content Generation, ready for n8n/orchestration."
APP_VERSION = "1.0.0"

# --- Scoring Heuristic Configuration (Day 3 Compliance) ---
# Defines the points awarded for the simple reward scoring system.

SCORING_RULES = {
    "structure_match": {
        "description": "Clear structure match (Introduction, Core Content, Takeaway, Why This Works).",
        "points": 10
    },
    "sentiment_alignment": {
        "description": "Sentiment/tone alignment with the Goal.",
        "points": 10
    },
    "feedback_inclusion": {
        "description": "Inclusion of previous feedback terms (if provided).",
        "points": 10
    },
    "readability": {
        "description": "Readability and flow.",
        "points": 5
    }
}

MAX_TOTAL_SCORE = sum(rule['points'] for rule in SCORING_RULES.values()) # Should equal 35

# --- Story Template Structure ---
# The required template structure to be enforced in the prompt.
UNIVERSAL_STORY_TEMPLATE = """
1. Introduction - Context and hook
2. Core Content - 3-5 plot beats or logical sections
3. Takeaway - Key insight or resolution
4. Why This Works - Creator note (hidden from viewer)
"""
