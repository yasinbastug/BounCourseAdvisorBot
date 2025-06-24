#!/usr/bin/env python
"""
Create a bigger, richer GEN training file (`data/gen_train.jsonl`)
by asking GPT-4o-mini to write *both* the Q&A and a short prose
"explanation" that we can train the LM to copy verbatim.

➜  export OPENAI_API_KEY=sk-…
➜  python synth_gpt4o.py --n 8          # 8 examples / course   (~3 K rows)
"""

from openai import OpenAI
from pymongo import MongoClient
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, json, os, random, time, backoff, config

# ------------------------------------------------------------------ cfg
N_PER_COURSE   = 8
OUT_FILE       = "data/gen_train.jsonl"
MAX_WORKERS    = 25

client   = OpenAI(api_key=config.OPENAI_API_KEY)                     # uses $OPENAI_API_KEY
mongo    = MongoClient(config.MONGO_URI)
courses  = list(mongo["course_advisor"]["courses"].find({}))

# ------------------------------------------------------------------ utils
QUESTION_TEMPL = {
    "name"  : [
        "What is the full title of the course {code}?",
        "Which course is identified by the code {code}?"
    ],
    "code"  : [
        "What is the course code for {name}?",
        "How is {name} listed in the catalogue?"
    ],
    "credit": [
        "How many credits is {name} worth?",
        "What are the credit hours for {code}?"
    ],
    "descr" : [
        "Give me a summary of what {name} covers.",
        "What topics are covered in {code}?"
    ]
}

def make_prompt(c):
    """Craft a *single* system+user message for GPT-4o-mini."""
    q_type   = random.choice(list(QUESTION_TEMPL))
    question = random.choice(QUESTION_TEMPL[q_type]).format(**c)

    system = (
      "You are an academic catalogue bot. "
      "Given the JSON, write **(1)** the same question, "
      "**(2)** ONE natural paraphrase, "
      "**(3)** a factual answer drawn *only* from the JSON, "
      "**(4)** a 1-2 sentence explanation (max 40 tokens) that also "
      "quotes at least one exact field. "
      "Return strict JSON with keys: question, paraphrase, answer, explanation."
    )
    user   = f"JSON: {json.dumps(c, ensure_ascii=False)}\n\nQuestion: {question}"
    return system, user

@backoff.on_exception(backoff.expo, Exception, max_tries=6)
def call_gpt(system, user):
    chat = client.chat.completions.create(
        model      = "gpt-4o-mini",
        temperature= 0.3,
        max_tokens = 200,
        messages   = [
            {"role":"system", "content":system},
            {"role":"user",   "content":user}
        ]
    )
    return chat.choices[0].message.content.strip()

def process_course(args):
    c, n = args
    results = []
    c = c.copy()
    c.pop("_id")
    c.pop("embedding")
    
    for _ in range(n):
        sysmsg, usrmsg = make_prompt(c)
        try:
            txt = call_gpt(sysmsg, usrmsg)
            obj = json.loads(txt)
            obj["context"] = c
            results.append(json.dumps(obj, ensure_ascii=False))
        except Exception as e:
            print(" !!", e)
            print(txt)
            continue
        time.sleep(0.3)
    return results

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_COURSE,
                    help="examples per course (default 8)")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_course, (c, args.n)) 
                for c in courses
            ]
            
            for future in tqdm(as_completed(futures), total=len(courses), desc="Generating"):
                for result in future.result():
                    fout.write(result + "\n")
                    
    print("Saved →", OUT_FILE)
