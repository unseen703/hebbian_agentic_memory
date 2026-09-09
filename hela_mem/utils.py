import time
import uuid
import numpy as np
from google import genai
from google.genai import types
import threading
import os
import json

# Process-level model cache
_model_cache = {}
_model_lock = threading.Lock()

# API Key Rotation System
_api_keys = None
_api_key_index = 0
_api_key_lock = threading.Lock()

# Default Google AI (Gemini API) model. Gemma is an open-weights family served
# through the same endpoint; override with HEBBIAN_MODEL.
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

_dotenv_loaded = False


def load_dotenv(path=".env"):
    """Load KEY=VALUE pairs from a .env file without overriding real env vars."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_api_keys():
    """Load Google AI API keys from env. Supports GOOGLE_AI_API_KEYS or GOOGLE_AI_API."""
    load_dotenv()

    keys_env = os.environ.get("GOOGLE_AI_API_KEYS", "").strip()
    if keys_env:
        keys = [key.strip() for key in keys_env.split(",") if key.strip()]
        if keys:
            return keys

    keys_file = os.environ.get("GOOGLE_AI_API_KEYS_FILE", "").strip()
    if keys_file:
        with open(keys_file, "r", encoding="utf-8") as f:
            keys = [line.strip() for line in f if line.strip()]
        if keys:
            return keys

    for var in ("GOOGLE_AI_API", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        single_key = os.environ.get(var, "").strip()
        if single_key:
            return [single_key]

    return []

def _get_next_api_key():
    """Get next API key in rotation (thread-safe)."""
    global _api_keys
    if _api_keys is None:
        _api_keys = load_api_keys()
    if not _api_keys:
        raise RuntimeError(
            "No API key configured. Set GOOGLE_AI_API or GOOGLE_AI_API_KEYS before running HeLa-Mem."
        )

    global _api_key_index
    with _api_key_lock:
        key = _api_keys[_api_key_index % len(_api_keys)]
        _api_key_index += 1
    return key

def _create_client(api_key=None):
    """Create a Google AI (Gemini API) client with given or rotated API key."""
    if api_key is None:
        api_key = _get_next_api_key()
    base_url = os.environ.get("GOOGLE_AI_BASE_URL", "").strip()
    http_options = types.HttpOptions(base_url=base_url) if base_url else None
    return genai.Client(api_key=api_key, http_options=http_options)


def split_messages(messages):
    """Convert OpenAI-style chat messages into (system_instruction, contents).

    System turns are concatenated into a single system instruction; the rest
    become google-genai Content turns ("assistant" maps to the "model" role).
    """
    system_parts = []
    contents = []
    for message in messages:
        role = message.get("role", "user")
        text = message.get("content", "")
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
            continue
        genai_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(role=genai_role, parts=[types.Part.from_text(text=text)])
        )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _generate(client, model, messages, temperature=0.7, max_tokens=2000):
    """Single Google AI generate_content call using OpenAI-style messages."""
    system_instruction, contents = split_messages(messages)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_tokens,
        # No tools are used; silences the SDK's per-call AFC advisory.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    text = getattr(response, "text", None)
    return text.strip() if text else None

try:
    llm_client = _create_client()
except RuntimeError:
    llm_client = None

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def get_embedding(text, model_name="all-MiniLM-L6-v2"):
    """
    Thread-safe, process-safe embedding generation
    """
    process_key = f"{os.getpid()}_{model_name}"

    if process_key not in _model_cache:
        with _model_lock:
            if process_key not in _model_cache:
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch

                    # Force CPU for stability in multiprocessing
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''

                    # Stagger model loading to prevent file system race conditions
                    import random
                    time.sleep(random.uniform(0.1, 5.0))

                    print(f"Process {os.getpid()} loading model: {model_name}")
                    model = SentenceTransformer(model_name, device='cpu')
                    model.eval()
                    # Warmup
                    with torch.no_grad():
                        _ = model.encode(['warmup'], convert_to_numpy=True, show_progress_bar=False)
                    _model_cache[process_key] = model
                except Exception as e:
                    print(f"Error loading model: {e}")
                    raise

    model = _model_cache[process_key]
    with _model_lock:
        embedding = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]

    return embedding

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def gpt_generate_answer(prompt, messages, client=None, model=None):
    # Use model from environment if not specified
    if model is None:
        model = os.environ.get('HEBBIAN_MODEL', DEFAULT_MODEL)
    # Use rotated API key for each call to reduce rate limits
    if client is None:
        client = _create_client()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            answer = _generate(client, model, messages, temperature=0.7, max_tokens=2000)

            if not answer:
                print(f"LLM Warning: Empty response. Attempt {attempt+1}/{max_retries}")
                time.sleep(2)
                continue

            return answer

        except Exception as e:
            print(f"LLM Error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            else:
                return ""
    return ""

def compute_time_decay(timestamp_str, tau=None):
    """简单的时间衰减函数"""
    from datetime import datetime
    import os

    # Read tau from environment variable if not provided
    if tau is None:
        tau = float(os.environ.get('HEBBIAN_TAU', '1e7'))

    try:
        t1 = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.now()
        delta = (t2 - t1).total_seconds()
        return np.exp(-delta/tau)
    except:
        return 1.0


def llm_extract_keywords(text, client=None):
    """
    Extract keywords from text using LLM.
    Added for Hebbian Memory improvement (Keyword Matching).
    """
    if client is None:
        client = llm_client
    if client is None:
        raise RuntimeError(
            "Keyword extraction requires LLM access. Set GOOGLE_AI_API or GOOGLE_AI_API_KEYS."
        )

    prompt = "Please extract the keywords of the conversation topic from the following dialogue, separated by commas, and do not exceed three:\n" + text
    messages = [
        {"role": "system", "content": "You are a keyword extraction expert. Please extract the keywords of the conversation topic."},
        {"role": "user", "content": prompt}
    ]

    keywords_text = gpt_generate_answer(prompt, messages, client)
    keywords = [w.strip() for w in keywords_text.split(",") if w.strip()]
    return set(keywords)


def gpt_generate_answer_with_rotation(prompt, messages, model=None, max_retries=3):
    """
    Generate answer using LLM with API key rotation.
    Thread-safe: creates a new client with rotated API key for each call.
    """
    # Use model from environment if not specified
    if model is None:
        model = os.environ.get('HEBBIAN_MODEL', DEFAULT_MODEL)

    client = _create_client()  # Gets next API key via rotation

    for attempt in range(max_retries):
        try:
            answer = _generate(client, model, messages, temperature=0.7, max_tokens=2000)

            if not answer:
                print(f"LLM Warning: Empty response. Attempt {attempt+1}/{max_retries}")
                time.sleep(2)
                continue

            return answer

        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str or 'limit' in error_str or 'resource_exhausted' in error_str or '429' in str(e):
                print(f"[API Rate Limit Warning] Too many requests! Waiting longer... (Attempt {attempt+1}/{max_retries})")
                time.sleep(10 * (attempt + 1))  # Wait longer for rate limits
            else:
                print(f"LLM Error (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    return ""
    return ""
