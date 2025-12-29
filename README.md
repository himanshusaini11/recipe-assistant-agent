# Recipe Assistant Agent

Implemented an AI agent to help users find and adapt recipes. Built with **FastAPI** and **OpenAI-compatible Tool Calling** (via OpenRouter).

- Agent tools:
  - `search_recipes(ingredients: list[str], cuisine: str | None)`
  - `get_recipe_details(recipe_id: str)`
  - `adjust_servings(recipe_id: str, servings: int)`
- FastAPI endpoint: `POST /chat`
- Session memory (dietary preferences & allergies) stored in a Python dict keyed by `session_id`
- Uses the provided `recipes.json` only (no external recipe APIs)

## 1) How to run locally?

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Set OpenRouter key
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
```

### Start API
```bash
uvicorn app.main:app --reload --port 8000
```

### Test
```bash
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"I have chicken, tomatoes, and garlic. What can I make?","session_id":"s1"}' | jq
```

## 2) Run with Docker

```bash
docker build -t recipe-agent .
docker run --rm -p 8000:8000 \
  -e OPENROUTER_API_KEY="sk-or-v1-..." \
  -e OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct" \
  recipe-agent
```

## 3) API

### POST /chat
Body:
```json
{ "message": "string", "session_id": "string" }
```

Response:
```json
{ "response": "string", "recipes": [ ... ] }
```

`recipes` is returned when the agent performs a recipe search.

## 4) Test scenarios

### Scenario 1: Basic Search
- `I have chicken, tomatoes, and garlic. What can I make?`
- Agent calls `search_recipes`

### Scenario 2: Dietary Preference Memory
- `I'm vegetarian`
- `What quick recipes do you suggest?`
- Agent remembers and filters to vegetarian/vegan recipes

### Scenario 3: Scaling
- `Show me the Greek Salad recipe`
- `I need it for 6 people`
- Agent calls `get_recipe_details` then `adjust_servings`

## Architecture

- Tools are implemented as pure Python functions over the in-memory recipe dataset.
- The agent uses OpenAI-style tool calling via the OpenRouter endpoint.
- Session memory is stored in `SESSION_MEMORY` dict (diet + allergies + avoid list) and applied inside tool functions.
- If `OPENROUTER_API_KEY` is not set, the service still runs using a deterministic fallback mode (so you can demo behavior without an LLM).

## Further improvements

- Better ingredient normalization (stemming, synonyms) and fuzzy matching.
- Add structured response schema for recipes and better error messaging in the UI layer.
