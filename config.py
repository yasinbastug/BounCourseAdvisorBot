# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
MONGO_URI = os.getenv('MONGO_URI', '')
DB_NAME = "course_advisor"
COURSE_COLL = "courses"

# Vector store configuration
FAISS_DIR = "faiss_store"
FAISS_INDEX_FILE = f"{FAISS_DIR}/courses.index"
FAISS_IDMAP_FILE = f"{FAISS_DIR}/courses_id_map.json"

# Model paths
KR_MODEL_PATH = "./kr_first_tuned"          # bi-encoder
RERANKER_NAME = "./reranker_tuned"     # instead of the HF base ckpt
GEN_MODEL_PATH = "./gen_finetuned"     # fine-tuned DialoGPT

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
