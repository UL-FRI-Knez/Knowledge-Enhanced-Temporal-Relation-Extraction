# import evaluate
import evaluate
import numpy as np
import pandas as pd
import seqeval
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification

from custom_datasets import combining_data
from custom_datasets.common import split_data
from custom_datasets.dataframe_dataset import DFDataset
from custom_datasets.event_extraction_dataset import convert_relation_extraction_df_to_event_extraction
from event_extraction.event_extraction_model import EventExtraction

# metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # return metric.compute(predictions=predictions, references=labels)
    return sum(labels == predictions) / len(predictions)

def combine_labels_lists_to_tensor(labels_lists):
    padded_lists = []
    length = max([len(x) for x in labels_lists])
    for l in labels_lists:
        padded_lists.append(l + [False] * (length - len(l)))
    return torch.tensor(padded_lists)

tokenizer = AutoTokenizer.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
def collate_fn(batch):
    texts = []
    tokens = []
    labels = []
    for example in batch:
        # texts.append(example["text"])
        tokens.append(example["tokens"])
        labels.append(example["labels"])
    return {"tokens": data_collator(tokens), "labels": combine_labels_lists_to_tensor(labels)}

def compute_metrics(p):
    predictions, labels = p
    raw_predictions = predictions[1]
    mask = predictions[3]
    pred = np.argmax(raw_predictions, axis=2)
    encoded_labels = [[1 if x else 0 for x in a] for a in labels]

    true_predictions = []
    for predL, labelsL, maskL in zip(pred, encoded_labels, mask):
        for p, l, m in zip(predL, labelsL, maskL):
            if m > 0:
                true_predictions.append(p == l)

    # results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # return {
    #     "precision": results["overall_precision"],
    #     "recall": results["overall_recall"],
    #     "f1": results["overall_f1"],
    #     "accuracy": results["overall_accuracy"],
    # }
    metrics = {"precision": sum(true_predictions) / len(true_predictions)}
    print(metrics)
    return metrics

def train_event_extraction():
    seqeval = evaluate.load("seqeval")

    df = combining_data.read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df_events = convert_relation_extraction_df_to_event_extraction(df, tokenizer)
    border1 = int(0.7 * len(df_events))
    border2 = int((0.7 + 0.2) * len(df_events))
    df_train, df_val, df_test = np.split(df_events, [border1, border2])

    dataset_train = DFDataset(df_train, lambda row, args: {"besedilo":row["text"], "labels":row["labels"], "tokens": row["tokens"]}, {})
    dataset_val = DFDataset(df_val, lambda row, args: {"besedilo":row["text"], "labels":row["labels"], "tokens": row["tokens"]}, {})

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    model = EventExtraction(tokenizer)
    model.to(device)
    # model = MultiModalPrediction(number_of_relations=3, combine_embeddings=True)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.01,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        auto_find_batch_size=True,
        num_train_epochs=50,
        weight_decay=0.001,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False,
    )
    # "adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision" or "adafactor"
    training_args.set_optimizer(name="adafactor")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    torch.save(model, "event-model.pt")
    pass

if __name__ == '__main__':
    train_event_extraction()