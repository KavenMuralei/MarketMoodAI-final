import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from huggingface_hub import notebook_login
notebook_login()

def preprocess_function(examples):
    return tokenizer(examples['headlines'], truncation=True)

def compute_metrics(eval_pred):
   metric = evaluate.load("accuracy")
   load_f1 = evaluate.load("f1")

   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = evaluate.load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

dataset = load_dataset("csv", data_files="./model_creation/all-data.csv")

# print(torch.cuda.is_available())
# print(dataset)
full_dataset = dataset["train"]

train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
small_train_dataset = train_test_split["train"].shuffle(seed=42).select(range(4360))
small_test_dataset = train_test_split["test"].shuffle(seed=42).select(range(436))

print(small_train_dataset.column_names)

# filepath: c:\projects\marketmoodai\final\model_creation\model.py
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")



tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

