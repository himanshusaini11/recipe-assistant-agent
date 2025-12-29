import os
import json
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError

logger = logging.getLogger("recipe_agent")

# Data loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def recipes_path() -> str:
    env_path = os.getenv("RECIPES_PATH")
    candidates = []
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        os.path.join(BASE_DIR, "recipes.json"),
        os.path.join(os.path.dirname(BASE_DIR), "recipes.json"),
        os.path.join(os.getcwd(), "recipes.json"),
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return env_path or "recipes.json"

RECIPES_PATH = recipes_path()

def load_recipes(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

DATA = load_recipes(RECIPES_PATH)
RECIPES = {r["id"]: r for r in DATA.get("recipes", [])}

# Session memory (in-memory dict)
# session_id -> preferences dict
SESSION_MEMORY: Dict[str, Dict[str, Any]] = {}

def get_session_prefs(session_id: str) -> Dict[str, Any]:
    return SESSION_MEMORY.setdefault(session_id, {
        "diet": None,               # e.g., "vegetarian", "vegan"
        "allergies": set(),         # e.g., {"peanuts", "shellfish"}
        "avoid_ingredients": set(), # additional avoids
    })

# Preference extraction (simple heuristics)
DIET_PATTERNS = {
    "vegan": re.compile(r"\b(vegan)\b", re.I),
    "vegetarian": re.compile(r"\b(vegetarian)\b", re.I),
}

ALLERGY_PATTERNS = [
    (re.compile(r"\b(allergic to|allergy|no)\s+(peanuts|peanut)\b", re.I), "peanuts"),
    (re.compile(r"\b(allergic to|allergy|no)\s+(shellfish|shrimp)\b", re.I), "shellfish"),
    (re.compile(r"\b(allergic to|allergy|no)\s+(dairy|milk|cheese)\b", re.I), "dairy"),
    (re.compile(r"\b(allergic to|allergy|no)\s+(eggs?)\b", re.I), "eggs"),
    (re.compile(r"\b(allergic to|allergy|no)\s+(gluten|wheat)\b", re.I), "gluten"),
]

def update_prefs_from_message(prefs: Dict[str, Any], message: str) -> None:
    # Diet
    for diet, pat in DIET_PATTERNS.items():
        if pat.search(message):
            prefs["diet"] = diet

    # Allergies / avoids
    for pat, item in ALLERGY_PATTERNS:
        if pat.search(message):
            prefs["allergies"].add(item)

    # Generic avoid: "avoid X" / "don't use X"
    m = re.findall(r"\b(avoid|don't use|do not use|no)\s+([a-zA-Z ]{2,30})\b", message, flags=re.I)
    for _, ing in m:
        ing = ing.strip().lower()
        if ing:
            prefs["avoid_ingredients"].add(ing)

# Tools
def normalize_ingredient(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def recipe_is_allowed(recipe: Dict[str, Any], prefs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Returns (allowed, reasons_if_blocked)
    """
    reasons = []
    tags = set(t.lower() for t in recipe.get("tags", []))
    ingredients = [normalize_ingredient(i["item"]) for i in recipe.get("ingredients", [])]

    diet = prefs.get("diet")
    if diet == "vegan":
        if "vegan" not in tags:
            reasons.append("Not tagged vegan")
    elif diet == "vegetarian":
        # vegetarian allows vegan or vegetarian
        if ("vegetarian" not in tags) and ("vegan" not in tags):
            reasons.append("Not tagged vegetarian/vegan")

    allergies = set(prefs.get("allergies", set()))
    avoid = set(prefs.get("avoid_ingredients", set()))

    # crude mapping for common allergens
    allergen_map = {
        "peanuts": {"peanuts", "peanut"},
        "shellfish": {"shrimp", "shellfish"},
        "dairy": {"milk", "cheese", "parmesan", "mozzarella", "feta cheese", "cheddar cheese", "butter", "heavy cream", "yogurt"},
        "eggs": {"egg", "eggs"},
        "gluten": {"bread", "flour", "spaghetti", "pasta", "taco shells"},
    }

    def contains_any(ings: List[str], needles: set) -> bool:
        for x in ings:
            if x in needles:
                return True
        return False

    for a in allergies:
        needles = allergen_map.get(a, {a})
        if contains_any(ingredients, set(normalize_ingredient(n) for n in needles)):
            reasons.append(f"Contains allergen: {a}")

    # avoid ingredients: check if any ingredient string contains avoid phrase
    for av in avoid:
        for ing in ingredients:
            if av in ing:
                reasons.append(f"Contains avoided ingredient: {av}")
                break

    return (len(reasons) == 0, reasons)

def search_recipes(ingredients: Optional[List[str]], cuisine: Optional[str], session_id: str) -> List[Dict[str, Any]]:
    prefs = get_session_prefs(session_id)
    ingredients = ingredients or []
    wanted = set(normalize_ingredient(x) for x in ingredients if x.strip())
    cuisine_norm = normalize_ingredient(cuisine) if cuisine else None

    results = []
    for r in RECIPES.values():
        if cuisine_norm and normalize_ingredient(r.get("cuisine", "")) != cuisine_norm:
            continue

        r_ings = [normalize_ingredient(i["item"]) for i in r.get("ingredients", [])]
        overlap = len(wanted.intersection(r_ings))
        if overlap == 0 and wanted:
            continue

        allowed, reasons = recipe_is_allowed(r, prefs)
        if not allowed:
            continue

        score = overlap / max(1, len(wanted)) if wanted else 0.0
        results.append({
            "id": r["id"],
            "name": r["name"],
            "cuisine": r.get("cuisine"),
            "prep_time": r.get("prep_time"),
            "servings": r.get("servings"),
            "tags": r.get("tags", []),
            "match_score": round(score, 3),
            "matched_ingredients": sorted(list(wanted.intersection(r_ings))),
        })

    results.sort(key=lambda x: (x["match_score"], -(x.get("prep_time") or 9999)), reverse=True)
    return results[:8]

def get_recipe_details(recipe_id: str, session_id: str) -> Dict[str, Any]:
    prefs = get_session_prefs(session_id)
    r = RECIPES.get(recipe_id)
    if not r:
        raise KeyError(f"Unknown recipe_id: {recipe_id}")

    allowed, reasons = recipe_is_allowed(r, prefs)
    if not allowed:
        raise PermissionError("Recipe conflicts with stored preferences: " + "; ".join(reasons))

    return r

def adjust_servings(recipe_id: str, servings: int, session_id: str) -> Dict[str, Any]:
    if servings <= 0:
        raise ValueError("servings must be positive")
    prefs = get_session_prefs(session_id)
    r = RECIPES.get(recipe_id)
    if not r:
        raise KeyError(f"Unknown recipe_id: {recipe_id}")

    allowed, reasons = recipe_is_allowed(r, prefs)
    if not allowed:
        raise PermissionError("Recipe conflicts with stored preferences: " + "; ".join(reasons))

    base_servings = r.get("servings", 1) or 1
    factor = servings / base_servings

    scaled = json.loads(json.dumps(r))  # deep copy
    scaled["servings"] = servings
    for ing in scaled.get("ingredients", []):
        amt = ing.get("amount")
        if isinstance(amt, (int, float)):
            ing["amount"] = round(amt * factor, 3)
    scaled["scaled_from_servings"] = base_servings
    scaled["scale_factor"] = round(factor, 3)
    return scaled

# LLM Agent (tool-using)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY or "missing",
)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": "Find matching recipes based on optional ingredients and optional cuisine. Applies saved dietary preferences for the session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredients": {"type": ["array", "null"], "items": {"type": "string"}},
                    "cuisine": {"type": ["string", "null"]},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recipe_details",
            "description": "Get full recipe including ingredients and steps for a recipe id. Applies saved dietary preferences for the session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_id": {"type": "string"},
                },
                "required": ["recipe_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_servings",
            "description": "Scale ingredient quantities for a recipe to a requested servings count. Applies saved dietary preferences for the session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_id": {"type": "string"},
                    "servings": {"type": "integer", "minimum": 1},
                },
                #"required": ["recipe_id", "servings"],
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a Recipe Assistant agent.
You MUST use the provided tools to:
- search recipes when the user asks what they can make, requests suggestions, or mentions ingredients.
- fetch details when the user asks to show a recipe, ingredients, steps, or wants a specific recipe by name.
- adjust servings when the user asks to scale/resize for N people/servings.
You must respect stored dietary preferences (vegan/vegetarian/allergies) which are handled by the tools.
If a recipe conflicts with preferences, explain and offer alternatives.
Return concise, user-friendly answers. When you recommend recipes, include a short list of matches with ids and prep time.
"""

def tool_dispatch(tool_name: str, args: Dict[str, Any], session_id: str) -> Any:
    if tool_name == "search_recipes":
        return search_recipes(args.get("ingredients"), args.get("cuisine"), session_id=session_id)
    if tool_name == "get_recipe_details":
        return get_recipe_details(args["recipe_id"], session_id=session_id)
    if tool_name == "adjust_servings":
        return adjust_servings(args["recipe_id"], int(args["servings"]), session_id=session_id)
    raise ValueError(f"Unknown tool: {tool_name}")

def run_agent(message: str, session_id: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    prefs = get_session_prefs(session_id)
    update_prefs_from_message(prefs, message)

    # Minimal short chat history per session for coherence
    history = prefs.setdefault("history", [])
    history.append({"role": "user", "content": message})
    history = history[-12:]  # cap

    # If no key, fall back to deterministic "rules" mode (still tool-based)
    if not OPENROUTER_API_KEY:
        return rules_fallback(message, session_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    # Tool-using loop
    recipes_payload = None
    for _ in range(4):
        try:
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2,
            )
        except OpenAIError as e:
            logger.warning("LLM call failed, falling back to rules mode: %s", e)
            fallback_text, recipes_payload = rules_fallback(message, session_id)
            note = "LLM unavailable; using offline mode."
            return f"{note}\n\n{fallback_text}", recipes_payload
        choice = resp.choices[0]
        msg = choice.message

        # If tool calls exist, execute them
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() for tc in tool_calls]})
            for tc in tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                try:
                    out = tool_dispatch(name, args, session_id=session_id)
                except Exception as e:
                    out = {"error": str(e)}
                # Keep track if it's a search result so API can return it
                if name == "search_recipes" and isinstance(out, list):
                    recipes_payload = out
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": json.dumps(out, ensure_ascii=False)})
            continue

        # Final assistant response
        final_text = msg.content or ""
        prefs["history"] = history + [{"role": "assistant", "content": final_text}]
        return final_text, recipes_payload

    return "I couldn't complete the request within tool steps. Please rephrase your request.", recipes_payload

def rules_fallback(message: str, session_id: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Offline fallback if no OpenRouter key is set. Still satisfies assignment behaviors.
    """
    msg = message.lower()

    # scaling intent
    m = re.search(r"\b(for|to)\s+(\d+)\s+(people|persons|servings?)\b", msg)
    if m:
        servings = int(m.group(2))
        # try to find recipe by name in message
        rid = find_recipe_id_by_name(message)
        if rid:
            scaled = adjust_servings(rid, servings, session_id=session_id)
            return format_recipe_details(scaled, include_scaled=True), None

    # show recipe intent
    if any(k in msg for k in ["show me", "recipe", "ingredients", "steps"]):
        rid = find_recipe_id_by_name(message)
        if rid:
            r = get_recipe_details(rid, session_id=session_id)
            return format_recipe_details(r), None

    # ingredient-based search intent
    if any(k in msg for k in ["i have", "what can i make", "suggest", "quick recipes"]):
        ings = extract_ingredients_simple(message)
        results = search_recipes(ings, None, session_id=session_id)
        return format_search_results(results, session_id), results

    # default: provide quick suggestion
    results = search_recipes([], None, session_id=session_id)
    return format_search_results(results, session_id), results

def find_recipe_id_by_name(text: str) -> Optional[str]:
    t = text.lower()
    # direct id mention
    m = re.search(r"\brecipe\s*#?\s*(\d+)\b", t)
    if m and m.group(1) in RECIPES:
        return m.group(1)
    # name match
    for rid, r in RECIPES.items():
        if r["name"].lower() in t:
            return rid
    # partial match by key words
    for rid, r in RECIPES.items():
        key = r["name"].lower().split("(")[0].strip()
        if key and key in t:
            return rid
    return None

def extract_ingredients_simple(text: str) -> List[str]:
    """
    Very simple ingredient extraction:
    - looks for "i have ..." then splits by commas/and
    - otherwise, attempts to match known ingredients from dataset
    """
    t = text.lower()
    m = re.search(r"\bi have\s+(.+)", t)
    if m:
        chunk = m.group(1)
        chunk = re.sub(r"[?.!]", "", chunk)
        parts = re.split(r",| and ", chunk)
        return [p.strip() for p in parts if p.strip()]

    # fallback: match any ingredient tokens in dataset
    all_ings = {normalize_ingredient(i["item"]) for r in RECIPES.values() for i in r.get("ingredients", [])}
    found = []
    for ing in sorted(all_ings, key=len, reverse=True):
        if ing and ing in t:
            found.append(ing)
    return list(dict.fromkeys(found))[:8]

def format_search_results(results: List[Dict[str, Any]], session_id: str) -> str:
    prefs = get_session_prefs(session_id)
    diet = prefs.get("diet")
    allergies = sorted(list(prefs.get("allergies", set())))
    header_bits = []
    if diet:
        header_bits.append(f"diet={diet}")
    if allergies:
        header_bits.append(f"allergies={', '.join(allergies)}")
    header = f"Preferences applied: {', '.join(header_bits)}.\n\n" if header_bits else ""

    if not results:
        return header + "No matching recipes found with the current constraints. Try adding more ingredients or removing cuisine constraints."

    lines = ["Here are some matches:"]
    for r in results:
        lines.append(f'- [{r["id"]}] {r["name"]} ({r["cuisine"]}, {r["prep_time"]} min) — matched: {", ".join(r["matched_ingredients"]) or "n/a"}')
    lines.append("\nTell me the recipe name or id to see details, or say “for N people” to scale it.")
    return header + "\n".join(lines)

def format_recipe_details(recipe: Dict[str, Any], include_scaled: bool = False) -> str:
    title = recipe.get("name", "Recipe")
    servings = recipe.get("servings", "")
    extra = ""
    if include_scaled and recipe.get("scale_factor"):
        extra = f' (scaled from {recipe.get("scaled_from_servings")} by {recipe.get("scale_factor")}x)'
    out = [f'{title} — Servings: {servings}{extra}', ""]
    out.append("Ingredients:")
    for ing in recipe.get("ingredients", []):
        out.append(f'- {ing.get("amount")} {ing.get("unit")} {ing.get("item")}')
    out.append("")
    out.append("Steps:")
    for i, step in enumerate(recipe.get("steps", []), 1):
        out.append(f"{i}. {step}")
    return "\n".join(out)

# FastAPI
app = FastAPI(title="Recipe Assistant Agent")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    response: str
    recipes: Optional[List[Dict[str, Any]]] = None

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    text, recipes = run_agent(req.message, req.session_id)
    return ChatResponse(response=text, recipes=recipes)

@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}
