# train_kr_from_scratch.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data      import DataLoader
import csv, tqdm, os

train_file = "nlp/app_project/data/kr_train.tsv"
model_name = "sentence-transformers/all-MiniLM-L6-v2"   # light & fast

# --- 3.1  Load base encoder
model = SentenceTransformer(model_name)

# --- 3.2  Build InputExample list
examples = []
with open(train_file, newline='', encoding='utf-8') as f:
    for q, doc in csv.reader(f, delimiter="\t"):
        examples.append(InputExample(texts=[q, doc]))

print(f"{len(examples):,} training pairs")

# --- 3.3  DataLoader
loader = DataLoader(examples, shuffle=True, batch_size=32)  # fit GPU vRAM

# --- 3.4  Contrastive loss (all other docs in batch = negatives)
loss_fn = losses.MultipleNegativesRankingLoss(model)

# --- 3.5  Fine-tune
model.fit(
    train_objectives   = [(loader, loss_fn)],
    epochs             = 3,
    warmup_steps       = int(0.1 * len(loader)),
    use_amp            = True,            # mixed precision
    output_path        = "./kr_first_tuned"
)

print("Model saved to ./kr_first_tuned")
