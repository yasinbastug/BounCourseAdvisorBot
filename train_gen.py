import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os
from typing import Dict, List
import logging
from torch.utils.data import Dataset as TorchDataset
import json
import requests
import zipfile
import io
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import wandb, but continue without it if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    logger.warning("wandb not available. Continuing without experiment tracking.")
    WANDB_AVAILABLE = False

class MultiWOZDataset(TorchDataset):
    def __init__(self, data_dir: str, split: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self._load_data(data_dir, split)
        
    def _load_data(self, data_dir: str, split: str) -> None:
        """Load and process MultiWOZ data."""
        split_dir = os.path.join(data_dir, split)
        for filename in os.listdir(split_dir):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(split_dir, filename)
            logger.info(f"Processing {filename}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                dialogues = json.load(f)
                
            for dialogue in dialogues:
                # Format dialogue turns
                formatted_dialogue = []
                for turn in dialogue:
                    # Each turn is a list with [user_utterance, system_utterance]
                    if turn[0]:  # User utterance
                        formatted_dialogue.append(f"User: {turn[0]}")
                    if turn[1]:  # System utterance
                        formatted_dialogue.append(f"Assistant: {turn[1]}")
                
                # Create training examples with context
                for i in range(1, len(formatted_dialogue)):
                    context = "\n".join(formatted_dialogue[:i])
                    response = formatted_dialogue[i]
                    self.examples.append({
                        'context': context,
                        'response': response
                    })
            
            logger.info(f"Processed {len(self.examples)} examples from {filename}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        
        # Combine context and response for training
        full_text = f"{example['context']}\n{example['response']}"
        
        # Tokenize the combined text
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze().clone()  # For causal language modeling
        }

def download_multiwoz():
    """Download and extract MultiWOZ dataset."""
    logger.info("Downloading MultiWOZ dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download MultiWOZ 2.2 from the official source
    url = "https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.2.zip?sequence=1&isAllowed=y"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Extract the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall('data/multiwoz_temp')
        
        # Move the data files to the correct location
        shutil.move('data/multiwoz_temp/data.json', 'data/multiwoz_train.json')
        shutil.move('data/multiwoz_temp/val_data.json', 'data/multiwoz_val.json')
        shutil.move('data/multiwoz_temp/test_data.json', 'data/multiwoz_test.json')
        
        # Clean up
        shutil.rmtree('data/multiwoz_temp')
        logger.info("MultiWOZ dataset downloaded and extracted successfully.")
    else:
        raise Exception(f"Failed to download MultiWOZ dataset. Status code: {response.status_code}. Please download the dataset manually from https://www.repository.cam.ac.uk/handle/1810/294507 and place it in the 'data' directory.")

def main():
    # Check if MultiWOZ dataset exists
    data_dir = 'data/MultiWOZ_2.2'
    if not os.path.exists(data_dir):
        raise Exception(f"MultiWOZ dataset not found at {data_dir}. Please ensure the dataset is downloaded and extracted correctly.")
    
    # Initialize model and tokenizer
    model_name = "microsoft/DialoGPT-medium"  # Using DialoGPT-medium as base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create custom datasets
    logger.info("Processing datasets...")
    train_dataset = MultiWOZDataset(data_dir, 'train', tokenizer)
    val_dataset = MultiWOZDataset(data_dir, 'dev', tokenizer)  # Using dev set for validation
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Initialize wandb if available
    if WANDB_AVAILABLE:
        try:
            wandb.init(project="course-advisor-bot", name="gen-pretraining")
            report_to = "wandb"
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}. Continuing without experiment tracking.")
            report_to = "none"
    else:
        report_to = "none"
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./gen_pretrained",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./gen_pretrained/logs",
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        report_to=report_to,  # Use wandb if available, otherwise none
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Keep eval_dataset for manual evaluation
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model("./gen_pretrained/final")
    tokenizer.save_pretrained("./gen_pretrained/final")
    
    # Close wandb if it was initialized
    if WANDB_AVAILABLE and report_to == "wandb":
        wandb.finish()
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 