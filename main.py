import os
import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure you have the following packages installed:
# pip install fastapi uvicorn pydantic langchain-google-genai google-genai

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# --- Pydantic Models for API Request/Response (Structured Data) ---

class StoryMetadata(BaseModel):
    """Structured filmmaking metadata required by the CreatorCore framework."""
    moodboard: List[str] = Field(description="Keywords for the visual mood of the story.")
    characters: List[Dict[str, str]] = Field(description="List of characters with 'name', 'trait', and 'role'.")
    scene: Dict[str, str] = Field(description="Scene details including 'location' and 'cinematography_note'.")
    time_of_day: str = Field(description="The time of day for the story's setting.")


class StoryRequest(BaseModel):
    """Input for generating the story script and metadata."""
    api_key: str = Field(..., description="Your Gemini API key.")
    topic: str = Field(..., description="The central topic for the content.")
    goal: str = Field(..., description="The objective of the content.")
    feedback: Optional[str] = Field(None,
                                    description="Previous feedback for script revision (used for iterative refinement).")


class StoryResponse(BaseModel):
    """Output containing the generated script and structured metadata."""
    story_script: str = Field(...,
                              description="The generated video script following the Universal Storytelling Template.")
    story_metadata: StoryMetadata = Field(..., description="Structured filmmaking metadata in JSON format.")
    raw_response: Dict[str, Any] = Field(..., description="Raw JSON output from the LLM call.")


class ScoreRequest(BaseModel):
    """Input for scoring a generated script."""
    api_key: str = Field(..., description="Your Gemini API key.")
    topic: str = Field(..., description="The central topic for the content.")
    goal: str = Field(..., description="The objective of the content.")
    story_script: str = Field(..., description="The story script to be scored.")
    feedback: Optional[str] = Field(None, description="The feedback that was used for regeneration, if any.")


class ScoreDetails(BaseModel):
    """Breakdown of the reward scoring heuristic."""
    structure_match: int = Field(description="+10 for clear structure match.")
    sentiment_alignment: int = Field(description="+10 for sentiment/tone alignment.")
    feedback_inclusion: int = Field(description="+10 for inclusion of feedback terms.")
    readability: int = Field(description="+5 for readability.")


class ScoreResponse(BaseModel):
    """Output for the reward score."""
    score_details: ScoreDetails = Field(..., description="Breakdown of the story's reward score.")
    total_score: int = Field(..., description="The total reward score for the story.")


class ThumbnailRequest(BaseModel):
    """Input for generating a thumbnail description."""
    api_key: str = Field(..., description="Your Gemini API key.")
    topic: str = Field(..., description="The central topic for the content.")
    goal: str = Field(..., description="The objective of the content.")
    story_metadata: StoryMetadata = Field(...,
                                          description="Structured filmmaking metadata from the story generation step.")
    feedback: Optional[str] = Field(None, description="Previous feedback for thumbnail revision.")


class ThumbnailResponse(BaseModel):
    """Output containing the thumbnail design description."""
    design_description: str = Field(..., description="A detailed text description for an image generation model.")


class MetadataRequest(BaseModel):
    """Input for generating final SEO metadata."""
    api_key: str = Field(..., description="Your Gemini API key.")
    story_script: str = Field(..., description="The final video script.")
    thumbnail_desc: str = Field(..., description="The final thumbnail design description.")


class MetadataResponse(BaseModel):
    """Output containing the final SEO-optimized video metadata."""
    video_title: str
    video_description: str
    tags: List[str]
    posting_time_recommendation: str
    target_audience: str
    raw_text: str = Field(..., description="The raw, formatted text output from the model for all metadata.")


# --- Helper Functions (LLM Configuration) ---

def initialize_llm(api_key: str, temperature: float = 0.7, model_name: str = "gemini-2.5-flash"):
    """Initializes the LangChain LLM and ensures the API key is used."""
    if not api_key:
        raise ValueError("API Key must be provided.")

    # Initialize LangChain model
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        # Set safety settings to be less restrictive as per new guidelines
        safety_settings=[
            HarmCategory.HARM_CATEGORY_HARASSMENT, HarmBlockThreshold.BLOCK_NONE
        ]
    )
    return llm


def get_gemini_model(api_key: str, model_name: str = 'gemini-2.5-pro'):
    """Initializes the raw Gemini GenerativeModel for structured output."""
    if not api_key:
        raise ValueError("API Key must be provided.")
    # Configure Gemini SDK to enable JSON mode and direct calls
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Warning: Gemini SDK configuration failed: {e}")
    return genai.GenerativeModel(model_name)


# --- CreatorCore Agent Functions (Core Business Logic) ---

def generate_story_and_metadata_core(req: StoryRequest) -> StoryResponse:
    """
    Generates a structured story script and filmmaking metadata in one go,
    enforcing the Universal Storytelling Template and structured JSON output (Day 2).
    """
    # Define the desired JSON structure
    json_schema = {
        "type": "object",
        "properties": {
            "story_script": {
                "type": "string",
                "description": "The full video script following the Universal Storytelling Template."
            },
            "story_metadata": StoryMetadata.model_json_schema()
        },
        "required": ["story_script", "story_metadata"]
    }

    template_structure = """
    1. Introduction - Context and hook
    2. Core Content - 3-5 plot beats or logical sections
    3. Takeaway - Key insight or resolution
    4. Why This Works - Creator note (hidden from viewer)
    """

    feedback_section = f"Previous script feedback to address: {req.feedback}" if req.feedback else "This is the initial script generation."

    prompt_template = f"""You are the CreatorCore Story Agent. Your task is to generate a complete video script and its structured filmmaking metadata.

{feedback_section}

Topic: {req.topic}
Goal: {req.goal}

The script must strictly follow the Universal Storytelling Template structure:
{template_structure}

The final output MUST be a single, valid JSON object that strictly adheres to the following schema. Do not include any text outside the JSON object.

JSON Schema: {json.dumps(json_schema)}

Please fill in the JSON keys:
- 'story_script': The actual script, following the template above.
- 'story_metadata': The structured filmmaking metadata, including moodboard, characters, scene, and time_of_day.
"""

    try:
        model = get_gemini_model(req.api_key, model_name='gemini-2.5-pro')
        response = model.generate_content(
            prompt_template,
            config={"response_mime_type": "application/json"}
        )

        raw_output_json = json.loads(response.text)

        # Validate and parse the output against Pydantic models
        story_metadata_data = raw_output_json.get("story_metadata", {})
        story_metadata = StoryMetadata(**story_metadata_data)

        return StoryResponse(
            story_script=raw_output_json.get("story_script", "Error: Script not found in JSON output."),
            story_metadata=story_metadata,
            raw_response=raw_output_json
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed structured generation: {e}")


def score_story_core(req: ScoreRequest) -> ScoreResponse:
    """
    Implements the basic reward scoring heuristic (Day 3) by using the LLM as a Scoring Agent.
    """

    scoring_rules_description = f"""
    - +10 for clear structure match (Introduction, Core Content, Takeaway, Why This Works).
    - +10 for sentiment/tone alignment with the Goal: '{req.goal}'.
    - +10 for inclusion of feedback terms (if feedback is provided: '{req.feedback or "None"}').
    - +5 for readability and flow.
    - Max total score: 35.
    """

    prompt_template = f"""You are the CreatorCore Scoring Agent. Analyze the following story script against the Topic, Goal, and Scoring Rules provided.

Topic: {req.topic}
Goal: {req.goal}
Feedback used for revision (if any): {req.feedback or "None"}

Story Script to Score:
---
{req.story_script}
---

Scoring Rules (Apply these strictly):
{scoring_rules_description}

Your output MUST be a single, valid JSON object that strictly adheres to the following structure. For each score category, you must output an integer score (0, 5, or 10 as specified by the rules).

JSON Schema for output:
{{
    "structure_match": [0 or 10],
    "sentiment_alignment": [0 or 10],
    "feedback_inclusion": [0 or 10],
    "readability": [0 or 5]
}}
"""

    try:
        model = get_gemini_model(req.api_key, model_name='gemini-2.5-pro')
        response = model.generate_content(
            prompt_template,
            config={"response_mime_type": "application/json"}
        )

        score_data = json.loads(response.text)

        # Calculate total score and validate
        total_score = sum(score_data.values())

        return ScoreResponse(
            score_details=ScoreDetails(**score_data),
            total_score=total_score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating score: {e}")


def generate_thumbnail_description_core(req: ThumbnailRequest) -> ThumbnailResponse:
    """Generates the text description for a thumbnail design, using structured metadata."""
    llm = initialize_llm(req.api_key, model_name="gemini-2.5-flash", temperature=0.8)

    feedback_section = f"Previous thumbnail feedback: {req.feedback}" if req.feedback else ""

    # Use structured metadata for a richer, more contextual prompt
    metadata_context = f"""
    --- Context from Story Metadata ---
    Mood: {', '.join(req.story_metadata.moodboard)}
    Main Characters: {', '.join([c['name'] for c in req.story_metadata.characters])}
    Scene Location/Note: {req.story_metadata.scene.get('location', 'N/A')} / {req.story_metadata.scene.get('cinematography_note', 'N/A')}
    Time of Day: {req.story_metadata.time_of_day}
    ---
    """

    prompt_template = f"""You are an expert visual designer. Create a compelling YouTube thumbnail image description.

Topic: {req.topic}
Goal: {req.goal}
{metadata_context}
{feedback_section}

Generate an eye-catching, professional, and high-contrast design description (for an image generation model) that stands out in search results and aligns with the story's creative context.

Thumbnail Design Description:"""

    prompt = PromptTemplate(input_variables=[], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        result = chain.run()
        return ThumbnailResponse(design_description=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating thumbnail description: {e}")


def generate_video_metadata_core(req: MetadataRequest) -> MetadataResponse:
    """Generates final video title, description, and tags."""
    llm = initialize_llm(req.api_key, model_name="gemini-2.5-pro", temperature=0.5)

    prompt_template = """Based on this script and thumbnail description, create full video metadata for YouTube optimization.

Script (first 500 chars): {script_snippet}...
Thumbnail Description: {thumbnail_desc_snippet}...

Provide the following in a structured text format:
1. Video Title (SEO optimized, under 60 characters)
2. Video Description (detailed, 200-300 words, including a strong call-to-action)
3. Tags (10-15 relevant tags, comma-separated)
4. Best posting time recommendation (e.g., 'Thursday at 3 PM EST')
5. Target audience description (e.g., 'Beginner Python developers, students')
"""

    script_snippet = req.story_script[:500]
    thumbnail_desc_snippet = req.thumbnail_desc[:200]

    prompt = PromptTemplate(
        input_variables=["script_snippet", "thumbnail_desc_snippet"],
        template=prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        raw_text_result = chain.run(
            script_snippet=script_snippet,
            thumbnail_desc_snippet=thumbnail_desc_snippet
        )

        # Simple parser to extract key fields
        def extract_field(text, prefix):
            line = next((line for line in text.split('\n') if line.startswith(prefix)), None)
            return line.split(': ', 1)[1].strip() if line and ': ' in line else "N/A"

        return MetadataResponse(
            video_title=extract_field(raw_text_result, '1. Video Title'),
            video_description=extract_field(raw_text_result, '2. Video Description'),
            tags=[t.strip() for t in extract_field(raw_text_result, '3. Tags').split(',') if t.strip()],
            posting_time_recommendation=extract_field(raw_text_result, '4. Best posting time recommendation'),
            target_audience=extract_field(raw_text_result, '5. Target audience description'),
            raw_text=raw_text_result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating video metadata: {e}")


# --- FastAPI App Initialization and Endpoints ---

app = FastAPI(
    title="CreatorCore Story Agent API",
    description="Modular backend for AI Content Generation, ready for n8n/orchestration (Day 1, 2, 3 compliant)."
)


# Root Endpoint for API Health Check
@app.get("/")
def read_root():
    return {"status": "ok", "service": "CreatorCore Story Agent API",
            "endpoints": ["/story/generate", "/story/score", "/thumbnail/generate", "/video/metadata"]}


# Story Generation Endpoint (Day 2: Template + Structured Metadata)
@app.post("/story/generate", response_model=StoryResponse,
          summary="Generate Story Script and Structured Filmmaking Metadata")
async def generate_story(req: StoryRequest):
    """
    Generates a video script following the **Universal Storytelling Template** and returns **structured JSON filmmaking metadata** (moodboard, characters, scene).
    """
    try:
        return generate_story_and_metadata_core(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# Story Scoring Endpoint (Day 3: Reward System)
@app.post("/story/score", response_model=ScoreResponse, summary="Score a Generated Story Script")
async def score_story(req: ScoreRequest):
    """
    Applies the **reward scoring heuristic** (+10 for structure, +10 for sentiment, +10 for feedback, +5 for readability)
    to a generated script and returns the total score and breakdown.
    """
    try:
        return score_story_core(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# Thumbnail Description Endpoint
@app.post("/thumbnail/generate", response_model=ThumbnailResponse, summary="Generate Thumbnail Design Description")
async def generate_thumbnail_desc(req: ThumbnailRequest):
    """
    Generates a text description for a thumbnail design, leveraging the **structured story metadata** for context.
    """
    try:
        return generate_thumbnail_description_core(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


# Video Metadata Endpoint
@app.post("/video/metadata", response_model=MetadataResponse, summary="Generate SEO-optimized Video Metadata")
async def generate_final_metadata(req: MetadataRequest):
    """
    Generates final SEO-optimized video metadata (Title, Description, Tags) based on the final script and thumbnail description.
    """
    try:
        return generate_video_metadata_core(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")