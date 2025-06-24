# gen_pretrain.py  (replace the preprocessing & Trainer part)

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import json, glob, os, torch

MODEL_NAME = "microsoft/DialoGPT-medium"
MAXLEN     = 1024

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:          # DialoGPT has no pad_token by default
    tok.pad_token = tok.eos_token

# ---------------------------------------------------------------------
# 1)  load MultiWOZ 2.2 from local JSON → list[dict]
def iter_dialogues(pattern):
    for fp in glob.glob(pattern):
        with open(fp, encoding="utf-8") as f:
            for dlg in json.load(f):
                # concatenate utterances with speaker tags – simple but ok
                turns = []
                for t in dlg["turns"]:
                    if t["speaker"] == "USER":
                        turns.append("User: " + t["utterance"])
                    else:
                        turns.append("Sys: "  + t["utterance"])
                yield {"text": "\n".join(turns)}

raw_ds = Dataset.from_list(list(iter_dialogues("data/MultiWOZ_2.2/train/*.json")))
print(raw_ds)

# ---------------------------------------------------------------------
# 2)  tokenize, pad, truncate, add labels
def tokenize_fn(batch):
    enc = tok(batch["text"],
              truncation=True,
              padding="max_length",
              max_length=MAXLEN)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tok_ds = raw_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
tok_ds.set_format(type="torch")            # tensors for Trainer
print(tok_ds[0].keys())                    # → dict with input_ids, attention_mask, labels

# ---------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

args = TrainingArguments(
    output_dir         = "./gen_pretrained",
    per_device_train_batch_size = 2,          # DialoGPT-medium is ~350M params
    gradient_accumulation_steps = 8,          # effective batch = 16
    num_train_epochs   = 1,
    learning_rate      = 5e-5,
    fp16               = torch.cuda.is_available(),
    logging_steps      = 100,
    save_steps         = 1000,
    remove_unused_columns = False,            # KEEP our input_ids / labels
)

data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

trainer = Trainer(
    model            = model,
    args             = args,
    train_dataset    = tok_ds,
    data_collator    = data_collator,
)

trainer.train()
trainer.save_model("./gen_pretrained")
tok.save_pretrained("./gen_pretrained")
print("GEN saved to ./gen_pretrained")
