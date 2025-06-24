# make_ce_data.py
import random, re, os, csv, itertools
from pathlib import Path

SRC = "data/cgda_pairs.txt"          # already created by embed phase
OUT = "data/ce_train.tsv"
os.makedirs("data", exist_ok=True)

pairs = []
with open(SRC, encoding="utf-8") as f:
    blob = f.read().strip().split("-"*80)
    for block in blob:
        q1 = re.search(r"Q1:(.*)", block)
        ans = re.search(r"A:(.*)",  block)
        if not (q1 and ans): continue
        pairs.append((q1.group(1).strip(), ans.group(1).strip()))

print(f"got {len(pairs)} Q/A pairs")

with open(OUT, "w", newline='', encoding="utf-8") as w:
    tsv = csv.writer(w, delimiter="\t")
    for q, pos in pairs:
        neg = random.choice([a for _, a in pairs if a != pos])   # cheap negative
        tsv.writerow([q, pos, neg])

print(f"wrote {OUT}")
