#!/usr/bin/env python
# finetune_gen.py ─ LoRA + unlikelihood anti-repetition
# =====================================================

import json, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

BASE_MODEL  = "microsoft/DialoGPT-medium"
DATA_FILE   = "data/gen_train.jsonl"
OUTPUT_DIR  = "gen_lora"
MAXLEN      = 512
ALPHA_UL    = 0.5                       # weight for unlikelihood loss

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# -------------------------------------------------------------------------
def make_prompt(question, ctx_obj, answer, expl=""):
    ctx_json = json.dumps(ctx_obj, ensure_ascii=False)
    prompt = (
        "Answer **only** from the JSON below.\n\n"
        f"JSON: {ctx_json}\n\n"
        f"Q: {question}\nA: {answer}"
    )
    if expl:
        prompt += "  " + expl            # optional explanation
    prompt += " <eos>"                   # explicit stop token
    return prompt

# ---- 1 · dataset ---------------------------------------------------------
ds = load_dataset("json", data_files=DATA_FILE, split="train")

def tok_fn(batch):
    prompts = [
        make_prompt(q, c, a, e if e else "")
        for q, c, a, e in zip(
            batch["question"],
            batch["context"],
            batch["answer"],
            batch.get("explanation", [""] * len(batch["question"]))
        )
    ]
    enc = tok(prompts, max_length=MAXLEN,
              truncation=True, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
ds.set_format("torch")

collator = DataCollatorForLanguageModeling(tok, mlm=False)

# ---- 2 · LoRA model ------------------------------------------------------
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg)

def unlikelihood_loss(log_probs: torch.Tensor, labels: torch.Tensor):
    # log_probs: (B, T, V), labels: (B, T)
    vocab_size = log_probs.size(-1)

    # build a mask of valid label positions
    valid_mask = (labels >= 0) & (labels < vocab_size)  # exclude -100 or PAD etc.

    # replace invalid labels with 0 so gather won't crash
    safe_labels = labels.masked_fill(~valid_mask, 0)

    # get the probability of the labeled token
    probs = log_probs.exp().gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

    # only keep the valid ones
    probs = probs[valid_mask]

    # unlikelihood = -log(1 - p)
    ul = -torch.log(torch.clamp(1.0 - probs, min=1e-8))

    return ul.mean()


class ULTrainer(Trainer):
    # accept *any* future kwargs → keeps us forward-compatible
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **unused,                   # ← absorbs num_items_in_batch
    ):
        labels = inputs.pop("labels")
        out = model(**inputs, labels=labels)
        ce = out.loss
        ul = unlikelihood_loss(out.logits, labels)
        loss = ce + ALPHA_UL * ul
        return (loss, out) if return_outputs else loss

# ---- 4 · training --------------------------------------------------------
args = TrainingArguments(
    OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    save_total_limit=2,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none",
)

trainer = ULTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collator,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print("✓ LoRA model with anti-repetition loss saved to", OUTPUT_DIR)
