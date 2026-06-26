
# farmlingua_backend/app/utils/config.py
from pathlib import Path
import os
import sys


BASE_DIR = Path(__file__).resolve().parents[2]


if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STATIC_VS_PATH = BASE_DIR / "app" / "vectorstore" / "faiss_index"
LIVE_VS_PATH = BASE_DIR / "app" / "vectorstore" / "live_rag_index"

VECTORSTORE_PATH = LIVE_VS_PATH 


WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "1eefcad138134d62a1e220003252608")


CLASSIFIER_PATH = BASE_DIR / "app" / "models" / "intent_classifier_v2.joblib"
CLASSIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.6"))


EXPERT_MODEL_NAME = os.getenv("EXPERT_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
#FORMATTER_MODEL_NAME = os.getenv("FORMATTER_MODEL_NAME", "google/flan-t5-large")

LANG_ID_MODEL_REPO = os.getenv("LANG_ID_MODEL_REPO", "facebook/fasttext-language-identification")
LANG_ID_MODEL_FILE = os.getenv("LANG_ID_MODEL_FILE", "model.bin")

TRANSLATION_MODEL_NAME = os.getenv("TRANSLATION_MODEL_NAME", "drrobot9/nllb-ig-yo-ha-finetuned")

DATA_SOURCES = {
    "harvestplus": "https://agronigeria.ng/category/news/",
}

STATES = [
    "Abuja", "Lagos", "Kano", "Kaduna", "Rivers", "Enugu", "Anambra", "Ogun",
    "Oyo", "Delta", "Edo", "Katsina", "Borno", "Benue", "Niger", "Plateau",
    "Bauchi", "Adamawa", "Cross River", "Akwa Ibom", "Ekiti", "Osun", "Ondo",
    "Imo", "Abia", "Ebonyi", "Taraba", "Kebbi", "Zamfara", "Yobe", "Gombe",
    "Sokoto", "Kogi", "Bayelsa", "Nasarawa", "Jigawa"
]


hf_cache = "/models/huggingface"
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = hf_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
os.makedirs(hf_cache, exist_ok=True)
