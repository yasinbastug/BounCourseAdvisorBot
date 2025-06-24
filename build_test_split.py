# build_test_split.py
import random, json
from pymongo import MongoClient
import config            # ← your existing config.py

TEST_COURSE_FRACTION = 0.10          # 10 % of all courses
OUT_FILE            = "data/test_courses.jsonl"

mongo = MongoClient(config.MONGO_URI)
courses = list(mongo["course_advisor"]["courses"].find({},
                {"embedding":0}))    # drop heavy vector

random.seed(42)
random.shuffle(courses)
n_test   = max(1, int(len(courses)*TEST_COURSE_FRACTION))
held_out = courses[:n_test]

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for c in held_out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"✓ Saved {n_test} held-out courses to {OUT_FILE}")
