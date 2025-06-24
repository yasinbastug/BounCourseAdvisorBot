# build_test_qa.py
import json, os, openai, tqdm, time
import config

openai.api_key = config.OPENAI_API_KEY
MODEL = "gpt-4o-mini"

def system_prompt(course):
    return (
      "You are a data-generation assistant. "
      "Given a JSON course record, create one factual question that a student "
      "MUST be able to answer using ONLY that record, and give the correct answer "
      "verbatim from the record.\n\n"
      f"JSON:\n{json.dumps(course, ensure_ascii=False)}\n\n"
      "Return *only* a JSON object with keys {question, answer}."
    )

in_path  = "data/test_courses.jsonl"
out_path = "data/test_qa.jsonl"

with open(in_path, encoding="utf-8") as f:
    courses = [json.loads(x) for x in f]

with open(out_path, "w", encoding="utf-8") as fout:
    for c in tqdm.tqdm(courses):
        while True:                        # simple retry loop
            try:
                resp = openai.chat.completions.create(
                    model=MODEL,
                    messages=[{"role":"system","content":system_prompt(c)}],
                    temperature=0.3,
                )
                qa = json.loads(resp.choices[0].message.content)
                qa["context"] = {k:c[k] for k in
                                 ("code","name","credits","ects","description",
                                  "prerequisites","department","faculty")}
                fout.write(json.dumps(qa, ensure_ascii=False)+"\n")
                time.sleep(0.3)
                break
            except Exception as e:
                print("Retrying:", e)
                time.sleep(2)
print(f"✓ Saved generated Q/A to {out_path}")
