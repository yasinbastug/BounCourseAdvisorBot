from datasets import Dataset
import json, glob

def load_local_multiwoz(split="train"):
    pattern = f"data/MultiWOZ_2.2/{split}/*.json"
    examples = []
    for fp in glob.glob(pattern):
        with open(fp, encoding="utf-8") as f:
            for dlg in json.load(f):
                utts = [t["utterance"] for t in dlg["turns"]]
                examples.append({"text": " \n ".join(utts)})
    return Dataset.from_list(examples)

ds = load_local_multiwoz("train")

print(ds)
# → Dataset({
#     features: ['text'],
#     num_rows: 70000   # or similar
# })

print(ds[0]["text"][:200], "...")
