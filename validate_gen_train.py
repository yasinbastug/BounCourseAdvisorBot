# validate_gen_train.py ------------------------------------------------
import json, pathlib, sys

SRC = pathlib.Path("data/gen_train.jsonl")
DST = pathlib.Path("data/gen_train.clean.jsonl")

bad, kept = 0, 0
with SRC.open(encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
    for ln, raw in enumerate(fin, 1):
        raw = raw.strip()
        if not raw:
            continue                              # skip blank lines
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[line {ln}] JSON error ⟶ {e}", file=sys.stderr)
            bad += 1
            continue

        # -------- schema normalisation ---------
        ans = obj.get("answer")
        # Arrow needs the column type to be constant → cast to string
        if isinstance(ans, dict):
            obj["answer"] = ans.get("text", json.dumps(ans, ensure_ascii=False))

        # (optional) drop unknown keys etc.

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print(f"✓ wrote {kept} lines → {DST}  |  {bad} lines dropped/errored")
