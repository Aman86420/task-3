CreatorCore Story Agent — README

A small FastAPI service that wraps Google’s Gemini (via the google-generativeai SDK and LangChain) to generate video scripts, structured filmmaking metadata, thumbnails, and SEO-optimized video metadata.

This README explains what the project does and gives step-by-step instructions to run the project locally and call each endpoint.

Features

/story/generate — Generate a full video script following a Universal Storytelling Template and return structured filmmaking metadata (moodboard, characters, scene, time_of_day).

/story/score — Score a provided script using a simple reward heuristic.

/thumbnail/generate — Produce a text description for a thumbnail design (suitable for image-generation models).

/video/metadata — Produce SEO-friendly title, description, tags, posting recommendation and target audience for a script.

All endpoints expect JSON requests and return structured JSON responses (Pydantic models are used for validation).

Prerequisites

Python 3.10+ (3.11 recommended)

pip

A Gemini / Google Generative API key (the project uses google-generativeai and langchain-google-genai). You need an API key from Google that allows calls to the Gemini models.

Files

main.py — FastAPI app with endpoints and core logic (you provided).

requirements.txt — Required packages (create from the earlier message or run pip install fastapi uvicorn ...).

If you don't have one, create requirements.txt with the packages suggested earlier:

fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
langchain==0.3.3
langchain-google-genai==1.0.4
google-generativeai==0.7.2
python-dotenv==1.0.1
typing-extensions>=4.8.0

Quick start — run locally (step-by-step)

Clone or place the project folder on your machine (containing main.py and requirements.txt).

Create and activate a virtual environment (recommended):

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1


Install dependencies:

pip install -r requirements.txt


Set your Gemini / Google Generative API key.

This project expects the API key to be provided in request bodies (the api_key field). However, some parts of the code also call genai.configure(api_key=...) internally. For convenience during development you can set an environment variable (optional):

# Linux / macOS
export GEMINI_API_KEY=""

# Windows (PowerShell)
$env:GEMINI_API_KEY=""


Note: The endpoints also accept api_key in the JSON body (this is required by the pydantic models in main.py). Supplying it in the request body is the straightforward approach.

Run the FastAPI server with Uvicorn:

uvicorn main:app --reload --port 8000


Open the interactive API docs in your browser:

http://127.0.0.1:8000/docs


You can test endpoints directly from the Swagger UI.

Example requests

Below are example curl requests you can use. Replace "YOUR_API_KEY_HERE" with your actual API key.

1) Generate story + metadata
curl -X POST "http://127.0.0.1:8000/story/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_API_KEY_HERE",
    "topic": "How to start a small kitchen garden",
    "goal": "Teach beginners the 5 essential steps and inspire them to start",
    "feedback": null
  }'


Response: a JSON object matching the StoryResponse model:

story_script (string)

story_metadata (object with moodboard, characters, scene, time_of_day)

raw_response (raw model output)

2) Score a script
curl -X POST "http://127.0.0.1:8000/story/score" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_API_KEY_HERE",
    "topic": "How to start a small kitchen garden",
    "goal": "Teach beginners the 5 essential steps and inspire them to start",
    "story_script": "PLACE_THE_SCRIPT_TEXT_HERE",
    "feedback": null
  }'


Response: ScoreResponse with score_details and total_score.

3) Generate thumbnail description
curl -X POST "http://127.0.0.1:8000/thumbnail/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_API_KEY_HERE",
    "topic": "How to start a small kitchen garden",
    "goal": "Teach beginners the 5 essential steps and inspire them to start",
    "story_metadata": {
      "moodboard": ["bright","organic","close-up plants"],
      "characters": [{"name":"Host","trait":"friendly","role":"narrator"}],
      "scene": {"location":"balcony","cinematography_note":"close-up shots of hands planting"},
      "time_of_day":"morning"
    },
    "feedback": null
  }'


Response: ThumbnailResponse containing a detailed description for a thumbnail image-generator.

4) Generate video metadata (title, description, tags)
curl -X POST "http://127.0.0.1:8000/video/metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_API_KEY_HERE",
    "story_script": "THE_FINAL_SCRIPT_TEXT_HERE",
    "thumbnail_desc": "Thumbnail description text..."
  }'


Response: MetadataResponse (title, description, tags, posting time recommendation, target audience, and raw_text).

Development notes & tips

API key handling: Current code expects api_key in the request body for each endpoint (as StoryRequest, ScoreRequest, etc.). If you prefer not to send the API key in the body, you can modify the code to read the key from an environment variable (e.g., os.getenv("GEMINI_API_KEY")) or implement a header-based approach (e.g., Authorization: Bearer <key>).

Model names & temperature: initialize_llm and get_gemini_model accept model names (e.g., "gemini-2.5-pro", "gemini-2.5-flash"). Change parameters in code if you want different models or temperature settings.

Error handling: The code wraps LLM calls in try/except and raises HTTPException on failure—check logs/traceback to debug model errors.

Rate limits & billing: Be mindful of API usage, quotas and costs when testing many requests.

Local testing: Use /docs for quick testing or tools like httpie, Postman, or Insomnia.

(Optional) Docker

A simple Dockerfile you can add (optional):

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


Build and run:

docker build -t creatorcore .
docker run -e GEMINI_API_KEY="YOUR_KEY" -p 8000:8000 creatorcore


You might still need to pass the api_key in request bodies unless you modify the code to read env var.

Troubleshooting

ImportError or missing package: make sure you installed the exact versions from requirements.txt.

Authentication or 403 from Google Generative API: confirm your API key is valid and has usage access for Gemini models.

JSON decode errors from LLM: the code expects certain response shapes — sometimes models return unexpected output. Check logs and the raw_response field returned in StoryResponse to inspect what the model returned.

Contributing / Next steps

Add auth middleware to accept API key via headers and avoid sending keys in bodies.

Add unit tests for the request/response flows (pytest, mocking model responses).

Add rate-limiting and request validation for production use.

Add a front-end UI to compose topics/feedback and display generated metadata.

License

Include your preferred license (e.g., MIT). Example short blurb:

MIT License
Copyright (c) 2025 <Your Name>


If you want, I can:

generate a README.md file and place it in the repo for you, or

create a .env.example with environment variable hints,

add Postman/HTTPie example files or a simple HTML UI to call the endpoints.

Which of those would you like next?
