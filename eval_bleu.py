# eval_bleu.py
import sacrebleu, json

REF_FILE = "data/test_qa.jsonl"
SYS_FILE = "data/sys_pred.txt"

refs = []
with open(REF_FILE, encoding="utf-8") as f:
    for l in f:
        refs.append(json.loads(l)["answer"])

with open(SYS_FILE, encoding="utf-8") as f:
    sys_out = [l.strip() for l in f]

bleu = sacrebleu.corpus_bleu(sys_out, [refs])
print(f"BLEU = {bleu.score:.2f}")

# Optional: ROUGE-L / METEOR
import evaluate
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
rouge_result = rouge.compute(predictions=sys_out, references=refs)
meteor_result = meteor.compute(predictions=sys_out, references=refs)
print(f"ROUGE-L: {rouge_result['rougeL'] * 100:.2f}")
print(f"METEOR : {meteor_result['meteor'] * 100:.2f}")
