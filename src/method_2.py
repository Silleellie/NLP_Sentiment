import re

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer, \
    TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import datasets

# metric = load_metric("accuracy")

class CustomDataset(torch.utils.data.Dataset):
    """
    https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def dataset_builder(raw_dataset_path):
    train = pd.read_csv(raw_dataset_path, sep="\t")
    all_texts = list(train['Phrase'])[:1000]
    all_labels = list(train['Sentiment'])[:1000]

    train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, train_size=.7)
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(train_texts, train_labels,
                                                                                      train_size=.9)

    train_dict = {'Phrase': train_texts, 'Sentiment': train_labels}
    validation_dict = {'Phrase': validation_texts, 'Sentiment': validation_labels}
    test_dict = {'Phrase': test_texts, 'Sentiment': test_labels}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    validation_dataset = datasets.Dataset.from_dict(validation_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                         "validation": validation_dataset,
                                         "test": test_dataset})

    return dataset_dict


def tokenize_fn(single_item_dataset):
    return tokenizer(single_item_dataset["Phrase"], truncation=True)

def right_attention(single_item_dataset):
    for i, token_id in enumerate(single_item_dataset['input_type_ids']):

        if token_id in special_tokens:
            single_item_dataset['attention_mask'][i] = 0

if __name__ == '__main__':

    dataset = dataset_builder('dataset/train.tsv')

    MODEL = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    dataset_tokenized = dataset.map(tokenize_fn, batched=True)

    special_tokens = set(tokenizer.all_special_ids)

    print(special_tokens)

    # model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    #
    # text = "Covid cases are increasing fast!"
    #
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    #
    # print(output)



def we():

    train = pd.read_csv("train.tsv", sep="\t")
    train_texts = list(train['Phrase'])[:1000]
    train_labels = list(train['Sentiment'])[:1000]

    train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

    args = TrainingArguments(
        output_dir="results_transformers",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    def model_init():
        return DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train()
    # We perform evaluation manually to obtain the predicted labels as output
    # trainer.evaluate()

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)
    output = pipe(test_texts)

    Y_pred = []

    for y in output:
        label = y['label']
        Y_pred.append(model.config.label2id[label])

    print(accuracy_score(test_labels, Y_pred))
