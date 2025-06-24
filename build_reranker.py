# build_reranker.py  (fixed)

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import csv, os

DATA = "data/ce_train.tsv"           # q \t pos_doc \t neg_doc
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 1) prepare pairs for CE using InputExample
train_samples = []
with open(DATA, encoding="utf-8") as f:
    tsv = csv.reader(f, delimiter="\t")
    for q, pos, neg in tsv:
        train_samples.append(InputExample(texts=[q, pos], label=1))
        train_samples.append(InputExample(texts=[q, neg], label=0))

# 2) split
train, dev = train_test_split(train_samples, test_size=0.05, random_state=42)

# 3) create DataLoaders
train_dataloader = DataLoader(train, shuffle=True, batch_size=16)
dev_dataloader = DataLoader(dev, shuffle=False, batch_size=32)

# 4) train CE
ce = CrossEncoder(MODEL, num_labels=1)

ce.fit(
    train_dataloader=train_dataloader,
    epochs=1,
    warmup_steps=100,
    evaluator=None,  # or you can add an evaluator here
    show_progress_bar=True,
)

# 5) save the model
ce.save_pretrained("./reranker_tuned")
print("saved to ./reranker_tuned")
