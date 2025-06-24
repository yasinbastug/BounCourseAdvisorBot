# run_system_on_test.py
import json, subprocess, tqdm

SRC   = "data/test_qa.jsonl"
PRED  = "data/sys_pred.txt"   # one answer per line

with open(SRC, encoding="utf-8") as f,         \
     open(PRED, "w", encoding="utf-8") as out:
    for line in tqdm.tqdm(f, total=sum(1 for _ in open(SRC,encoding="utf-8"))):
        qa = json.loads(line)
        q  = qa["question"]
        # call your CLI script and capture stdout
        result = subprocess.check_output(
            ["python", "gen_answer.py", q], text=True
        )
        # your script prints "A: …" – extract after that
        ans = result.split("A: ",1)[-1].strip()
        out.write(ans+"\n")
print(f"✓ System answers saved to {PRED}")
