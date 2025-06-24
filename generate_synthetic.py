# generate_synthetic.py
"""
Call GPT-4o-mini to create two paraphrased questions + an answer
for each course in Mongo.  Output → data/gen_train.jsonl
"""

import os, json, time, random
from bson import ObjectId
from pymongo import MongoClient
from tqdm.auto import tqdm
from openai import OpenAI
from config import MONGO_URI
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_FILE   = "data/gen_train.jsonl"
N_PER_DOC  = 2                 # how many Q-A triples per course
MODEL      = "gpt-4o-mini"
MAX_WORKERS = 10               # number of parallel workers

# Initialize OpenAI client with explicit API key handling
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)
mongo = MongoClient(MONGO_URI)
coll = mongo["course_advisor"]["courses"]

SYSTEM_MSG = (
    "You are generating synthetic training data for a university-course Q-A bot. "
    "For every course JSON I give you, produce **exactly** N pairs of distinct "
    "questions that ask for the same fact(s) about the course and one factual answer. "
    "Return a JSON list of objects with keys question1, question2, answer as: "
    "[{question1: '...', question2: '...', answer: '...'}]"
)

def build_prompt(course:dict, n:int)->str:
    course = {k:v for k,v in course.items()
              if not isinstance(v, (bytes, bytearray, ObjectId))}
    return (
        f"{SYSTEM_MSG}\n\n"
        f"N = {n}\n\n"
        f"Course JSON:\n{json.dumps(course, ensure_ascii=False, indent=2)}"
    )

def query_gpt(prompt:str, retries:int=5):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model   = MODEL,
                messages= [ {"role":"system", "content":SYSTEM_MSG},
                            {"role":"user",   "content":prompt} ],
                temperature = 0.7,
                max_tokens   = 512
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2**attempt + random.random()
            print(f"OpenAI error ({e}), retrying in {wait:.1f}s…")
            time.sleep(wait)
    raise RuntimeError("OpenAI failed too many times")

def process_doc(doc):
    results = []
    prompt = build_prompt(doc, N_PER_DOC)
    out = query_gpt(prompt)
    try:
        triples = json.loads(out)
        assert isinstance(triples, list)
    except (json.JSONDecodeError, AssertionError):
        if out.startswith('```json') and out.endswith('```'):
            try:
                out = out.split('```json')[1].split('```')[0]
                out = out.strip()
                triples = json.loads(out)
                assert isinstance(triples, list)
            except (json.JSONDecodeError, AssertionError):
                print("⚠️  Bad JSON from GPT, skipping")
                print(out)
                return results
        else:
            print("⚠️  Bad JSON from GPT, skipping")
            print(out)
            return results

    for t in triples:
        record = {
            "question" : t["question1"],
            "paraphrase": t["question2"],
            "context"  : {k:v for k,v in doc.items()
                          if k not in ("_id", "embedding")},
            "answer"   : t["answer"]
        }
        results.append(record)
    return results

# ----------------------------------------------------------------------

os.makedirs("data", exist_ok=True)
written = 0

with open(OUT_FILE, "w", encoding="utf-8") as fout:
    docs = list(coll.find({}, projection={"embedding":0}))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_doc = {executor.submit(process_doc, doc): doc for doc in docs}
        
        for future in tqdm(as_completed(future_to_doc), total=len(docs)):
            results = future.result()
            for record in results:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

print("Saved", written, "examples →", OUT_FILE)
