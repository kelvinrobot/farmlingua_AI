# farmlingua_backend/app/tasks/rag_updater.py
import os
import sys
from datetime import datetime, date
import logging
import requests
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils import config  

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

session = requests.Session()

def fetch_weather_now():
    """Fetch current weather for all configured states."""
    docs = []
    for state in config.STATES:
        try:
            url = "http://api.weatherapi.com/v1/current.json"
            params = {
                "key": config.WEATHER_API_KEY,
                "q": f"{state}, Nigeria",  
                "aqi": "no"
            }
            res = session.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()

            if "current" in data:
                condition = data['current']['condition']['text']
                temp_c = data['current']['temp_c']
                humidity = data['current']['humidity']
                text = (
                    f"Weather in {state}: {condition}, "
                    f"Temperature: {temp_c}°C, Humidity: {humidity}%"
                )
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": "WeatherAPI",
                        "location": state,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ))
        except Exception as e:
            logging.error(f"Weather fetch failed for {state}: {e}")
    return docs

def fetch_harvestplus_articles():
    """Fetch ALL today's articles from HarvestPlus site."""
    try:
        res = session.get(config.DATA_SOURCES["harvestplus"], timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("article")

        docs = []
        today_str = date.today().strftime("%Y-%m-%d")

        for a in articles:
            content = a.get_text(strip=True)
            if content and len(content) > 100:
                
                if today_str in a.text or True:  
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": "HarvestPlus",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    ))
        return docs
    except Exception as e:
        logging.error(f"HarvestPlus fetch failed: {e}")
        return []

def build_rag_vectorstore(reset=False):
    job_type = "FULL REBUILD" if reset else "INCREMENTAL UPDATE"
    logging.info(f"RAG update started — {job_type}")

    all_docs = fetch_weather_now() + fetch_harvestplus_articles()

    logging.info(f"Weather docs fetched: {len([d for d in all_docs if d.metadata['source'] == 'WeatherAPI'])}")
    logging.info(f"News docs fetched: {len([d for d in all_docs if d.metadata['source'] == 'HarvestPlus'])}")

    if not all_docs:
        logging.warning("No documents fetched, skipping update")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(all_docs)

    embedder = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)

    vectorstore_path = config.LIVE_VS_PATH  

    if reset and os.path.exists(vectorstore_path):
        for file in os.listdir(vectorstore_path):
            file_path = os.path.join(vectorstore_path, file)
            try:
                os.remove(file_path)
                logging.info(f"Deleted old file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}: {e}")

    if os.path.exists(vectorstore_path) and not reset:
        vs = FAISS.load_local(
            vectorstore_path,
            embedder,
            allow_dangerous_deserialization=True
        )
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embedder)

    os.makedirs(vectorstore_path, exist_ok=True)
    vs.save_local(vectorstore_path)

    logging.info(f"Vectorstore updated at {vectorstore_path}")

def schedule_updates():
    scheduler = BackgroundScheduler()
    scheduler.add_job(build_rag_vectorstore, 'interval', hours=12, kwargs={"reset": False})
    scheduler.add_job(build_rag_vectorstore, 'interval', days=7, kwargs={"reset": True})
    scheduler.start()
    logging.info("Scheduler started — 12-hour incremental updates + weekly full rebuild")
    return scheduler