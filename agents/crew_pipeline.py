# farmlingua/app/agents/crew_pipeline.pymemorysection
import os
import sys
import re
import uuid
import requests
import joblib
import faiss
import numpy as np
import torch
import fasttext
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from app.utils import config
from app.utils.memory import memory_store  # memory module
from typing import List


hf_cache = "/models/huggingface"
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = hf_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
os.makedirs(hf_cache, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


try:
    classifier = joblib.load(config.CLASSIFIER_PATH)
except Exception:
    classifier = None


print(f"Loading expert model ({config.EXPERT_MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(config.EXPERT_MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    config.EXPERT_MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


embedder = SentenceTransformer(config.EMBEDDING_MODEL)

#   language detector
print(f"Loading FastText language identifier ({config.LANG_ID_MODEL_REPO})...")
lang_model_path = hf_hub_download(
    repo_id=config.LANG_ID_MODEL_REPO,
    filename=getattr(config, "LANG_ID_MODEL_FILE", "model.bin")
)
lang_identifier = fasttext.load_model(lang_model_path)

def detect_language(text: str, top_k: int = 1):
    if not text or not text.strip():
        return [("eng_Latn", 1.0)]
    clean_text = text.replace("\n", " ").strip()
    labels, probs = lang_identifier.predict(clean_text, k=top_k)
    return [(l.replace("__label__", ""), float(p)) for l, p in zip(labels, probs)]

#  Translation model
print(f"Loading translation model ({config.TRANSLATION_MODEL_NAME})...")
translation_pipeline = pipeline(
    "translation",
    model=config.TRANSLATION_MODEL_NAME,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=400,
)

SUPPORTED_LANGS = {
    "eng_Latn": "English",
    "ibo_Latn": "Igbo",
    "yor_Latn": "Yoruba",
    "hau_Latn": "Hausa",
    "swh_Latn": "Swahili",
    "amh_Latn": "Amharic",
}

# Text chunking
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def chunk_text(text: str, max_len: int = 400) -> List[str]:
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks, current = [], ""
    for s in sentences:
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_len:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current.strip())
            current = s.strip()
    if current:
        chunks.append(current.strip())
    return chunks

def translate_text(text: str, src_lang: str, tgt_lang: str, max_chunk_len: int = 400) -> str:
    if not text.strip():
        return text
    chunks = chunk_text(text, max_len=max_chunk_len)
    translated_parts = []
    for chunk in chunks:
        res = translation_pipeline(chunk, src_lang=src_lang, tgt_lang=tgt_lang)
        translated_parts.append(res[0]["translation_text"])
    return " ".join(translated_parts).strip()

#  RAG retrieval
def retrieve_docs(query: str, vs_path: str):
    if not vs_path or not os.path.exists(vs_path):
        return None
    try:
        index = faiss.read_index(str(vs_path))
    except Exception:
        return None
    query_vec = np.array([embedder.encode(query)], dtype=np.float32)
    D, I = index.search(query_vec, k=3)
    if D[0][0] == 0:
        return None
    meta_path = str(vs_path) + "_meta.npy"
    if os.path.exists(meta_path):
        metadata = np.load(meta_path, allow_pickle=True).item()
        docs = [metadata.get(str(idx), "") for idx in I[0] if str(idx) in metadata]
        docs = [d for d in docs if d]
        return "\n\n".join(docs) if docs else None
    return None


def get_weather(state_name: str) -> str:
    url = "http://api.weatherapi.com/v1/current.json"
    params = {"key": config.WEATHER_API_KEY, "q": f"{state_name}, Nigeria", "aqi": "no"}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return f"Unable to retrieve weather for {state_name}."
    data = r.json()
    return (
        f"Weather in {state_name}:\n"
        f"- Condition: {data['current']['condition']['text']}\n"
        f"- Temperature: {data['current']['temp_c']}°C\n"
        f"- Humidity: {data['current']['humidity']}%\n"
        f"- Wind: {data['current']['wind_kph']} kph"
    )


def detect_intent(query: str):
    q_lower = (query or "").lower()
    if any(word in q_lower for word in ["weather", "temperature", "rain", "forecast"]):
        for state in getattr(config, "STATES", []):
            if state.lower() in q_lower:
                return "weather", state
        return "weather", None

    if any(word in q_lower for word in ["latest", "update", "breaking", "news", "current", "predict"]):
        return "live_update", None

    if hasattr(classifier, "predict") and hasattr(classifier, "predict_proba"):
        try:
            predicted_intent = classifier.predict([query])[0]
            confidence = max(classifier.predict_proba([query])[0])
            if confidence < getattr(config, "CLASSIFIER_CONFIDENCE_THRESHOLD", 0.6):
                return "low_confidence", None
            return predicted_intent, None
        except Exception:
            pass
    return "normal", None

# expert runner
def run_qwen(messages: List[dict], max_new_tokens: int = 1300) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.4,
        repetition_penalty=1.1
    )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

#  Memory
MAX_HISTORY_MESSAGES = getattr(config, "MAX_HISTORY_MESSAGES", 30)

def build_messages_from_history(history: List[dict], system_prompt: str) -> List[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history)
    return msgs


def strip_markdown(text: str) -> str:
    """
    Remove Markdown formatting like **bold**, *italic*, and `inline code`.
    """
    if not text:
        return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    return text

#  Main pipeline
def run_pipeline(user_query: str, session_id: str = None):
    """
    Run FarmLingua pipeline with per-session memory.
    Each session_id keeps its own history.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())  # fallback unique session

   
    lang_label, prob = detect_language(user_query, top_k=1)[0]
    if lang_label not in SUPPORTED_LANGS:
        lang_label = "eng_Latn"

    translated_query = (
        translate_text(user_query, src_lang=lang_label, tgt_lang="eng_Latn")
        if lang_label != "eng_Latn"
        else user_query
    )

    intent, extra = detect_intent(translated_query)

    #  Load conversation history
    history = memory_store.get_history(session_id) or []
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    
    history.append({"role": "user", "content": translated_query})

    
    system_prompt = (
        "You are FarmLingua, an AI assistant for Nigerian farmers. "
        "Answer directly without repeating the question. "
        "Use clear farmer-friendly English with emojis . "
        "Avoid jargon and irrelevant details. "
        "If asked who built you, say: 'KawaFarm LTD developed me to help farmers.'"
      
    )

    
    if intent == "weather" and extra:
        weather_text = get_weather(extra)
        history.append({"role": "user", "content": f"Rewrite this weather update simply for farmers:\n{weather_text}"})
        messages_for_qwen = build_messages_from_history(history, system_prompt)
        english_answer = run_qwen(messages_for_qwen, max_new_tokens=256)
    else:
        if intent == "live_update":
            context = retrieve_docs(translated_query, config.LIVE_VS_PATH)
            if context:
                history.append({"role": "user", "content": f"Latest agricultural updates:\n{context}"})
        if intent == "low_confidence":
            context = retrieve_docs(translated_query, config.STATIC_VS_PATH)
            if context:
                history.append({"role": "user", "content": f"Reference information:\n{context}"})

        messages_for_qwen = build_messages_from_history(history, system_prompt)
        english_answer = run_qwen(messages_for_qwen, max_new_tokens=700)

   
    history.append({"role": "assistant", "content": english_answer})
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
    memory_store.save_history(session_id, history)

    # Translate back if needed
    final_answer = (
        translate_text(english_answer, src_lang="eng_Latn", tgt_lang=lang_label)
        if lang_label != "eng_Latn"
        else english_answer
    )
    final_answer = strip_markdown(final_answer)
    return {
        "session_id": session_id,
        "detected_language": SUPPORTED_LANGS.get(lang_label, "Unknown"),
        "answer": final_answer
    }
